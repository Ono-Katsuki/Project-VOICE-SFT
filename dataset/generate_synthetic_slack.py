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
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from google import genai
except ImportError:
    print("pip install google-genai が必要です", file=sys.stderr)
    sys.exit(1)

# -----------------------------
# Model / Config
# -----------------------------
MODEL_GEN = "gemini-2.5-flash"
NO_THINKING_CONFIG = {"thinking_config": {"thinking_budget": 0}}

MAX_GEN_RETRIES = 3  # 生成がNGのときの再試行回数（generate_one内）
MAX_WORKER_RETRIES = 3  # API例外時のワーカー側再試行回数（レート制限等）

# -----------------------------
# Exhaustive lists (given)
# -----------------------------
SELF_ROLES = ["テックリーダー", "マネジメント", "エンジニア", "ビジネス"]
OTHER_ROLES = ["上司", "部下", "取引先", "お客様", "ビジネスパートナー", "同僚", "リーダー", "メンバー"]
RELATIONS = ["親しい", "仲が悪い", "要注意な", "頼みやすい", "頼みにくい", "優しい", "厳しい", "疎遠な"]
CONTENTS = ["肯定", "否定", "提案", "依頼", "催促", "報告", "相談", "連絡"]
SCENES = ["ビジネス", "開発", "マネジメント", "プレゼン"]

# -----------------------------
# Prompt
# -----------------------------
GEN_PROMPT = """あなたは社内/社外のSlackで使われる、日本語の短い発言（1〜3文）を作るアシスタントです。

【目的】
与えられる条件（自分の役割・相手・関係性・内容・シーン）に合う、自然なSlackメッセージ本文を1つ生成してください。

【厳守】
- 日本語のSlackメッセージとして自然な1〜3文（長文禁止、箇条書き禁止、長い前置き禁止）
- たとえ関係性が「仲が悪い」「厳しい」「要注意」でも、暴言・侮辱・差別・攻撃は絶対にしない（冷たい/事務的/距離感のある表現は可）
- 実在の個人名・会社名・住所・電話・メール・URL・アカウントIDなどの個人情報や固有名詞は出さない
- 記号や絵文字は使ってもよいが、使いすぎない
- 生成するのは「本文(text)」のみ

【出力フォーマット】
- 次の JSON を <GEN> と </GEN> の間に「1行だけ」出力。
- <GEN> タグの外側には何も出力しない。

<GEN>
{"text":"..."}
</GEN>
"""

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

def make_stable_id(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

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

def content_hint(scene: str, content: str) -> str:
    base = {
        "ビジネス": "商談/見積/契約/日程調整/要件確認などの文脈",
        "開発": "不具合/PR/レビュー/デプロイ/仕様確認/タスクなどの文脈",
        "マネジメント": "1on1/進捗/優先度/リソース/方針などの文脈",
        "プレゼン": "資料/スライド/リハ/想定QA/構成などの文脈",
    }.get(scene, "一般的な業務文脈")

    intent = {
        "肯定": "相手の意見や提案に賛成・了承する",
        "否定": "懸念点を述べて難しい/見送り/違うと伝える（丁寧に）",
        "提案": "代案や進め方を提案する",
        "依頼": "作業や確認をお願いする",
        "催促": "期限や返答をやんわり促す",
        "報告": "進捗や結果を共有する",
        "相談": "判断や進め方を相談する",
        "連絡": "共有事項や周知を伝える",
    }.get(content, "適切に意図を表現する")

    return f"{base}。内容は「{intent}」に寄せる。"

def relation_tone_hint(relation: str) -> str:
    mapping = {
        "親しい": "少しカジュアル寄りでもよいが礼節は保つ",
        "仲が悪い": "事務的・短め・余計な感情を入れない",
        "要注意な": "慎重・丁寧・誤解を招かない言い回し",
        "頼みやすい": "率直にお願いしてよい",
        "頼みにくい": "クッション言葉を入れて丁寧にお願いする",
        "優しい": "柔らかく感謝を入れる",
        "厳しい": "結論先出し・端的・曖昧にしない（攻撃しない）",
        "疎遠な": "丁寧・状況説明を短く添える",
    }
    return mapping.get(relation, "丁寧で自然なトーン")

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

def evaluate_text(text: str) -> Tuple[bool, str]:
    sc = sentence_count_ja(text)
    if sc < 1 or sc > 3:
        return False, f"文数が1〜3文ではありません: {sc}"
    if len(text) < 6:
        return False, "短すぎます"
    if len(text) > 220:
        return False, "長すぎます（Slack想定で上限超え）"
    if "\n\n" in text:
        return False, "空行があり、箇条書き/長文っぽい"
    if has_pii_like(text):
        return False, "PII/URL/メール/電話っぽいパターンが含まれています"
    return True, ""

def build_user_input(meta: Dict[str, str], style_seed: str) -> str:
    hints = [
        f"自分の役割: {meta['self_role']}",
        f"相手: {meta['other_role']}",
        f"相手との関係性: {meta['relation']}（トーン: {relation_tone_hint(meta['relation'])}）",
        f"内容: {meta['content']}",
        f"シーン: {meta['scene']}（話題の寄せ: {content_hint(meta['scene'], meta['content'])}）",
        f"バリエーション種: {style_seed}",
        "注意: 固有名詞・個人情報は出さない。1〜3文。Slackの本文だけ。",
    ]
    return "\n".join(hints)

# -----------------------------
# One record generation (NO fallback)
# -----------------------------
def generate_one(client: genai.Client, meta: Dict[str, str], retries: int = MAX_GEN_RETRIES) -> Dict[str, Any]:
    style_seed = meta.get("style_seed", "A")
    feedback = ""
    last_full = ""

    for attempt in range(retries + 1):
        user_input = build_user_input(meta, style_seed)
        instr = GEN_PROMPT
        if feedback:
            instr += (
                "\n\n【前回の出力が条件を満たしませんでした。必ず修正してください】\n"
                f"{feedback}\n"
            )

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

        ok_eval, reason_e = evaluate_text(text)
        if ok_eval:
            rid = meta["id"]
            return {
                "id": rid,
                "original_id": rid,
                "text": text,
                "self_role": meta["self_role"],
                "other_role": meta["other_role"],
                "relation": meta["relation"],
                "content": meta["content"],
                "scene": meta["scene"],
                "variant": meta["variant"],
                "style_seed": style_seed,
                "model": MODEL_GEN,
                "key": meta["key"],  # 再開用
            }

        feedback = (
            f"生成文が条件NGです: {reason_e}\n"
            "- 1〜3文に収める\n"
            "- 長すぎない\n"
            "- PII/URL/メール/電話は入れない\n"
            "- 箇条書きにしない\n"
            "を満たしてください。"
        )

    # fallbackしない：失敗として扱う
    raise RuntimeError(f"generate_one failed after retries. last_output={last_full[:300]!r}")

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
# Worker wrapper
# -----------------------------
def worker_generate_with_backoff(client: genai.Client, meta: Dict[str, str], gen_retries: int) -> Dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        try:
            return generate_one(client, meta, retries=gen_retries)
        except Exception as e:
            rate = is_rate_error(e)
            if rate:
                backoff = 0.6 * (2 ** min(attempt, 6)) + random.uniform(0, 0.5)
                time.sleep(backoff)
                continue

            # 非レート系は一定回数で諦めて上位に投げる（failed.jsonlへ）
            if gen_retries == -1:
                # 無限生成モードなら、非レートでも少し待って継続
                time.sleep(0.5)
                continue

            if attempt >= MAX_WORKER_RETRIES:
                raise
            time.sleep(0.3 * attempt)

# -----------------------------
# Main
# -----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="dataset/cleaned.jsonl")
    ap.add_argument("--failed-output", default="dataset/failed.jsonl")
    ap.add_argument("--variants", type=int, default=1, help="各組合せあたりのバリエーション数")
    ap.add_argument("--limit", type=int, default=None, help="生成件数の上限（デバッグ用）")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=3, help="-1 で無限リトライ（全組合せを埋めたい場合）")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    outp = Path(args.output)
    failp = Path(args.failed_output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    failp.parent.mkdir(parents=True, exist_ok=True)

    # 途中再開：既に出力済みの key を読む
    processed_keys = set()
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
                        processed_keys.add(k)
                except Exception:
                    continue

    metas: List[Dict[str, str]] = []
    style_seeds = ["A", "B", "C", "D", "E", "F"]

    for self_role in SELF_ROLES:
        for other_role in OTHER_ROLES:
            for relation in RELATIONS:
                for content in CONTENTS:
                    for scene in SCENES:
                        for v in range(args.variants):
                            key = f"{self_role}|{other_role}|{relation}|{content}|{scene}|v{v}"
                            if key in processed_keys:
                                continue
                            rid = make_stable_id(key)
                            metas.append({
                                "id": rid,
                                "key": key,
                                "self_role": self_role,
                                "other_role": other_role,
                                "relation": relation,
                                "content": content,
                                "scene": scene,
                                "variant": v,
                                "style_seed": style_seeds[v % len(style_seeds)],
                            })
                            if args.limit and len(metas) >= args.limit:
                                break
                        if args.limit and len(metas) >= args.limit:
                            break
                    if args.limit and len(metas) >= args.limit:
                        break
                if args.limit and len(metas) >= args.limit:
                    break
            if args.limit and len(metas) >= args.limit:
                break
        if args.limit and len(metas) >= args.limit:
            break

    if args.shuffle:
        random.shuffle(metas)

    total = len(metas)
    print(f"[INFO] to_generate={total} (variants={args.variants}) output={outp} failed={failp}", file=sys.stderr)
    if total == 0:
        print("no new records.")
        return

    client = get_client()
    lock = threading.Lock()

    done = 0
    failed = 0
    print_progress(done, total)

    with outp.open("a", encoding="utf-8") as outf, failp.open("a", encoding="utf-8") as ferr:
        with ThreadPoolExecutor(max_workers=min(args.max_workers, 50)) as ex:
            futures = [ex.submit(worker_generate_with_backoff, client, m, args.retries) for m in metas]

            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    rec = fut.result()
                    with lock:
                        outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        outf.flush()
                except Exception as e:
                    # fallbackしない：失敗は別ファイルに記録して clean から除外
                    failed += 1
                    # 対応するmetaは取り出せないので、例外情報のみ書くのは弱い → future順なので完全一致はできない
                    # そこで fut に紐づく meta を保持したい場合は、submit時に meta をクロージャに含める等に変更してください。
                    # ここでは簡易的にエラーだけ残す。
                    err = {"error": str(e)}
                    with lock:
                        ferr.write(json.dumps(err, ensure_ascii=False) + "\n")
                        ferr.flush()

                done += 1
                print_progress(done, total)

    print(f"done -> {outp} ({total - failed} records), failed -> {failp} ({failed} records)")

if __name__ == "__main__":
    main()
