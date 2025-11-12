#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, sys, time, threading, random
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from google import genai
except ImportError:
    print("pip install google-genai が必要です", file=sys.stderr)
    sys.exit(1)

# --- 変更点 1: モデル名を flash に変更 ---
MODEL_CLEAN = "gemini-2.5-flash"
MODEL_META  = "gemini-2.5-flash"

# --- 変更点 2: 思考なし設定を定義 ---
# (辞書として直接渡すため、インポートは不要)
NO_THINKING_CONFIG = {"thinking_config": {"thinking_budget": 0}}

CLEAN_PROMPT = """あなたはテキストを最小限にクリーンするアシスタントです。
以下のルールで入力テキストを「できるだけ元の言い回しを残して」書き換えてください。

【置換ルール】
- 実在しそうな人名 → 「<NAME>」
- メールアドレス → 「<EMAIL>」
- 電話番号・携帯番号 → 「<PHONE>」
- パスワード・トークン・APIキー・シークレットらしきもの → 「<SECRET>」
- 社名・組織名など明らかに固有のもの → 「<ORG>」
- 住所・場所の特定情報 → 「<ADDR>」
- URL は残してよいが、クエリに個人情報が含まれた場合は 「<URL>」 にする

【メール由来ぽい場合】
- 署名ブロック（" -- "や「よろしくお願いいたします。」以降の所属・電話・住所）は削除
- 過去メールの引用（"On ... wrote:" や "-----Original Message-----"）は削除

【重要】
- 元の文体・語順はできるだけ保つ
- 不明なものを一律にマスクしない
- 出力はクリーン後のテキストのみ
"""

META_PROMPT = """あなたはクリーン済みテキストに属性をつけるアシスタントです。
次の3項目をJSONで1行で出力してください。

【clean_score（1〜5, 整数）】
1: 個人情報・秘密情報・セキュリティリスクがほぼ無い
2: ごく軽微な固有情報が残っている可能性
3: 氏名や組織名など再識別に使われる情報が残る
4: 連絡先・固有名・組織名などが複数残る
5: クレデンシャルや明確な個人特定情報が残る

【language】主要言語
【genre】["メール","メッセージ","メモ","プログラム","仕様","その他"] のいずれか

例:
{"clean_score": 2, "language": "日本語", "genre": "メール"}
"""

def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("環境変数 GOOGLE_API_KEY がありません", file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=api_key)

def call_clean(client: genai.Client, text: str, source_kind: str) -> str:
    prompt = CLEAN_PROMPT
    if source_kind in ("gmail","threads_full","email","mail"):
        prompt += "\nこのテキストはメール由来です。\n"
    resp = client.models.generate_content(
        model=MODEL_CLEAN,
        contents=[prompt, text],
        config=NO_THINKING_CONFIG          # --- 変更点 3: 思考なし設定を適用 ---
    )
    return (resp.text or "").strip()

def call_meta(client: genai.Client, cleaned_text: str) -> Dict[str, Any]:
    resp = client.models.generate_content(
        model=MODEL_META,
        contents=[META_PROMPT, cleaned_text],
        config=NO_THINKING_CONFIG          # --- 変更点 4: 思考なし設定を適用 ---
    )
    raw = (resp.text or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"clean_score": 3, "language": "unknown", "genre": "その他"}

def is_rate_error(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "rate" in msg.lower() or "503" in msg

def process_one(shared_client: genai.Client, rec: Dict[str,Any], retries: int = 3, base_sleep: float = 0.5) -> Dict[str,Any]:
    """
    retries == -1 の場合は無限リトライ
    """
    attempt = 0
    last_err = None
    while True:
        attempt += 1
        try:
            cleaned = call_clean(shared_client, rec["text"], rec.get("source_kind") or "")
            meta = call_meta(shared_client, cleaned)
            return {
                "id": rec["clean_id"],
                "original_id": rec["original_id"],  # 文字列に正規化済み
                "text": cleaned,
                "clean_score": int(meta.get("clean_score", 3)),
                "language": meta.get("language", "unknown"),
                "genre": meta.get("genre", "その他"),
            }
        except Exception as e:
            last_err = e
            # レート系なら指数バックオフ
            if is_rate_error(e):
                backoff = base_sleep * (2 ** min(attempt, 6))  # 上限気持ち6段
                backoff = backoff + random.uniform(0, 0.5)
                time.sleep(backoff)
            else:
                # 非レートエラーは指定回数で打ち切り
                if retries == -1:
                    time.sleep(base_sleep)
                    continue
                if attempt >= retries:
                    break
                time.sleep(base_sleep * attempt)
        # 通常リトライ判定
        if retries != -1 and attempt >= retries:
            break

    return {
        "id": rec["clean_id"],
        "original_id": rec["original_id"],  # 文字列に正規化済み
        "text": rec["text"],
        "clean_score": 5,
        "language": "unknown",
        "genre": "その他",
        "error": str(last_err) if last_err else "unknown error"
    }

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

# --- 型のみ汎用化のための最小ヘルパ（original_id を文字列化） ---
def _to_id_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset/base_merged.jsonl")
    ap.add_argument("--output", default="dataset/cleaned.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3, help="-1 で無限リトライ")
    args = ap.parse_args()

    # ユーザー指定を尊重して50まで許可
    max_workers = min(args.max_workers, 50)

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str,Any]] = []
    clean_id = 1
    with inp.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src = json.loads(line)
            records.append({
                "clean_id": clean_id,
                "original_id": _to_id_str(src.get("id")),  # ★ ここだけ型汎用化（文字列正規化）
                "text": src.get("text") or "",
                "source_kind": src.get("source_kind") or "",
            })
            clean_id += 1
            if args.limit and len(records) >= args.limit:
                break

    total = len(records)
    if total == 0:
        print("no records.")
        return

    outf = outp.open("a", encoding="utf-8")
    lock = threading.Lock()

    client = get_client()

    done_count = 0
    print_progress(done_count, total)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_one, client, rec, args.retries)
            for rec in records
        ]
        for fut in as_completed(futures):
            result = fut.result()
            with lock:
                outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                outf.flush()
            done_count += 1
            print_progress(done_count, total)

    outf.close()
    print(f"done -> {outp} ({total} records)")

if __name__ == "__main__":
    main()
