#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import random
import re
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from google import genai
except ImportError:
    print("pip install google-genai が必要です", file=sys.stderr)
    sys.exit(1)

MODEL_GEN = "gemini-2.5-flash"
NO_THINKING_CONFIG = {"thinking_config": {"thinking_budget": 0}}

# -----------------------------
# Utilities
# -----------------------------
def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("環境変数 GOOGLE_API_KEY もしくは GEMINI_API_KEY がありません", file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=api_key)

def is_rate_error(e: Exception) -> bool:
    msg = str(e)
    return ("429" in msg) or ("rate" in msg.lower()) or ("503" in msg)

def sha_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def sentence_count_ja(text: str) -> int:
    t = re.sub(r"\s+", " ", text.strip())
    if not t:
        return 0
    parts = re.split(r"[。！？!?]+", t)
    parts = [p.strip() for p in parts if p.strip()]
    return max(1, len(parts))

def has_pii_like(text: str) -> bool:
    patterns = [
        r"https?://", r"www\.",
        r"[\w\.-]+@[\w\.-]+\.\w+",
        r"\b0\d{1,4}-\d{1,4}-\d{3,4}\b",
        r"\b\d{2,4}-\d{2,4}-\d{3,4}\b",
    ]
    return any(re.search(p, text) for p in patterns)

def evaluate_reply(text: str) -> Tuple[bool, str]:
    sc = sentence_count_ja(text)
    if sc < 1 or sc > 3:
        return False, f"文数が1〜3文ではありません: {sc}"
    if len(text) < 4:
        return False, "短すぎます"
    if len(text) > 220:
        return False, "長すぎます"
    if "\n" in text:
        return False, "改行が含まれています"
    if has_pii_like(text):
        return False, "PII/URL/メール/電話っぽいパターンが含まれています"
    return True, ""

def relation_tone_hint(relation: str) -> str:
    mapping = {
        "親しい": "少しカジュアル寄りでもよいが礼節は保つ",
        "仲が悪い": "事務的・短め・余計な感情を入れない（攻撃しない）",
        "要注意な": "慎重・丁寧・誤解を招かない言い回し",
        "頼みやすい": "率直に返答してよい",
        "頼みにくい": "クッション言葉を入れて丁寧に返答する",
        "優しい": "柔らかく感謝を入れる",
        "厳しい": "結論先出し・端的（ただし攻撃しない）",
        "疎遠な": "丁寧・状況説明を短く添える",
    }
    return mapping.get(relation, "丁寧で自然なトーン")

def reply_strategy_hint(content: str) -> str:
    # 元発言(content)に対して、返信でよくある動きをガイド
    return {
        "依頼": "依頼に対して、了承して対応予定を返すか、追加確認を1点だけ尋ねる",
        "催促": "催促に対して、状況説明＋いつまでに返すか（短く）",
        "報告": "報告に対して、受領/把握の返事＋必要なら次アクションを一言",
        "相談": "相談に対して、意見を返すか、判断材料を1点だけ確認する",
        "提案": "提案に対して、賛同/懸念＋次の進め方を一言",
        "肯定": "肯定に対して、了承＋次の一手を短く",
        "否定": "否定に対して、受け止め＋代案/条件確認を短く",
        "連絡": "連絡に対して、了解＋必要なら確認を一言",
    }.get(content, "自然な受け答えをする")

# -----------------------------
# Prompt / Parse
# -----------------------------
REPLY_PROMPT = """あなたは社内/社外のSlack返信（日本語・1〜3文）を作るアシスタントです。

【タスク】
与えられる文脈と「元メッセージ」に対して、相手側として自然な返信本文を1つ生成してください。

【厳守】
- 返信は1〜3文。長文・箇条書き・改行は禁止
- 元メッセージのコピペや引用返信（">" など）はしない
- たとえ関係性が悪くても、暴言・侮辱・差別・攻撃はしない（事務的/距離感は可）
- 実在の個人名・会社名・住所・電話・メール・URL・アカウントIDなどの個人情報や固有名詞は出さない
- 返信の本文(text)のみを生成

【出力フォーマット】
<GEN>
{"text":"..."}
</GEN>
"""

def parse_gen(full: str) -> Tuple[bool, str, str]:
    m = re.search(r"<GEN>\s*(\{.*?\})\s*</GEN>", full, re.DOTALL | re.IGNORECASE)
    raw = m.group(1).strip() if m else full.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        text = str(obj.get("text", "")).strip()
        if not text:
            return False, "", "text が空です"
        return True, text, ""
    except Exception as e:
        return False, "", f"JSONとして解釈できません: {e}"

# -----------------------------
# Thread-local client
# -----------------------------
_thread_local = threading.local()
def get_thread_client() -> genai.Client:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = get_client()
    return _thread_local.client

# -----------------------------
# Generation
# -----------------------------
def build_reply_input(src: Dict[str, Any], style_seed: str) -> str:
    # 元データ: src は slack.jsonl の1行
    # src["self_role"] が送信者、src["other_role"] が受信者（=返信者）
    sender = src.get("self_role", "")
    replier = src.get("other_role", "")
    relation = src.get("relation", "")
    scene = src.get("scene", "")
    content = src.get("content", "")
    parent_text = src.get("text", "")

    return "\n".join([
        f"元メッセージ送信者の役割: {sender}",
        f"返信者（あなた）の役割: {replier}",
        f"関係性: {relation}（トーン: {relation_tone_hint(relation)}）",
        f"シーン: {scene}",
        f"元メッセージの意図カテゴリ: {content}",
        f"返信方針ヒント: {reply_strategy_hint(content)}",
        f"バリエーション種: {style_seed}",
        "",
        "元メッセージ:",
        parent_text,
    ])

def generate_reply_one(src: Dict[str, Any], retries: int) -> Dict[str, Any]:
    style_seed = src.get("style_seed", "A")
    feedback = ""
    last_full = ""

    attempt = 0
    while True:
        attempt += 1
        if retries != -1 and attempt > (retries + 1):
            raise RuntimeError(f"generate_reply_one failed after retries. last_output={last_full[:300]!r}")

        client = get_thread_client()
        user_input = build_reply_input(src, style_seed)

        instr = REPLY_PROMPT
        if feedback:
            instr += "\n\n【前回の出力が条件を満たしませんでした。必ず修正してください】\n" + feedback + "\n"

        resp = client.models.generate_content(
            model=MODEL_GEN,
            contents=[instr, user_input],
            config=NO_THINKING_CONFIG,
        )
        full = (resp.text or "").strip()
        last_full = full

        ok_parse, text, reason_p = parse_gen(full)
        if not ok_parse:
            feedback = (
                "出力が指定JSON形式ではありません。<GEN>タグ内に1行JSONで "
                '{"text":"..."} を出してください。'
                f"\nエラー: {reason_p}"
            )
            continue

        ok_eval, reason_e = evaluate_reply(text)
        if ok_eval:
            parent_id = src.get("id") or src.get("original_id") or ""
            reply_key = (src.get("key") or "") + "|reply"
            rid = sha_id(reply_key)

            return {
                "id": rid,
                "original_id": rid,
                "text": text,                 # ← 返信本文（segmentationにそのまま流せる）
                "key": reply_key,
                "model": MODEL_GEN,

                # ペア復元用
                "parent_id": parent_id,
                "parent_key": src.get("key"),
                "parent_text": src.get("text"),

                # 文脈（返信者視点に入れ替えたメタも保持）
                "self_role": src.get("other_role"),   # 返信者（=相手）
                "other_role": src.get("self_role"),   # 返信先（=元送信者）
                "relation": src.get("relation"),
                "scene": src.get("scene"),
                "parent_content": src.get("content"),
            }

        feedback = (
            f"返信文が条件NGです: {reason_e}\n"
            "- 1〜3文\n- 改行なし\n- 引用しない\n- PIIなし\n- 長すぎない\n"
            "を満たしてください。"
        )

def worker(src: Dict[str, Any], retries: int) -> Dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        try:
            return generate_reply_one(src, retries=retries)
        except Exception as e:
            if is_rate_error(e):
                backoff = 0.6 * (2 ** min(attempt, 6)) + random.uniform(0, 0.5)
                time.sleep(backoff)
                continue

            if retries == -1:
                time.sleep(0.5)
                continue

            # 非レート系は少しだけ再挑戦してダメなら上に投げる
            if attempt >= 3:
                raise
            time.sleep(0.3 * attempt)

# -----------------------------
# Progress
# -----------------------------
def print_progress(done: int, total: int):
    width = 50
    ratio = done / total if total else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(ratio * 100)
    sys.stdout.write(f"\r[{bar}] {percent:3d}% ({done}/{total})")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")

# -----------------------------
# Main
# -----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="slack.jsonl")
    ap.add_argument("--output", default="slack_replies.jsonl")
    ap.add_argument("--failed-output", default="slack_replies_failed.jsonl")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=3, help="-1で無限リトライ")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    failp = Path(args.failed_output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    failp.parent.mkdir(parents=True, exist_ok=True)

    # 入力読み込み
    srcs: List[Dict[str, Any]] = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            srcs.append(json.loads(line))
            if args.limit and len(srcs) >= args.limit:
                break

    # 途中再開：既に reply_key を出力済みならスキップ
    done_keys = set()
    if outp.exists():
        with outp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    k = obj.get("key")
                    if k:
                        done_keys.add(k)
                except Exception:
                    continue

    targets = []
    for s in srcs:
        rk = (s.get("key") or "") + "|reply"
        if rk in done_keys:
            continue
        targets.append(s)

    total = len(targets)
    print(f"[INFO] targets={total} input={inp} output={outp} failed={failp}", file=sys.stderr)
    if total == 0:
        print("no new records.")
        return

    lock = threading.Lock()
    done = 0
    failed = 0
    print_progress(done, total)

    future_to_src = {}
    with outp.open("a", encoding="utf-8") as outf, failp.open("a", encoding="utf-8") as ferr:
        with ThreadPoolExecutor(max_workers=min(args.max_workers, 50)) as ex:
            for s in targets:
                fut = ex.submit(worker, s, args.retries)
                future_to_src[fut] = s

            for fut in as_completed(future_to_src):
                s = future_to_src[fut]
                try:
                    rec = fut.result()
                    with lock:
                        outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        outf.flush()
                except Exception as e:
                    failed += 1
                    err = {
                        "parent_id": s.get("id") or s.get("original_id"),
                        "parent_key": s.get("key"),
                        "parent_text": s.get("text"),
                        "error": str(e),
                    }
                    with lock:
                        ferr.write(json.dumps(err, ensure_ascii=False) + "\n")
                        ferr.flush()

                done += 1
                print_progress(done, total)

    print(f"done -> {outp} ({total - failed} records), failed -> {failp} ({failed} records)")

if __name__ == "__main__":
    main()
