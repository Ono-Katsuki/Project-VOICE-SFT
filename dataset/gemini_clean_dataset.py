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
        config=NO_THINKING_CONFIG
    )
    return (resp.text or "").strip()

def call_meta(client: genai.Client, cleaned_text: str) -> Dict[str, Any]:
    resp = client.models.generate_content(
        model=MODEL_META,
        contents=[META_PROMPT, cleaned_text],
        config=NO_THINKING_CONFIG
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
                "original_id": rec["original_id"],
                "text": cleaned,
                "clean_score": int(meta.get("clean_score", 3)),
                "language": meta.get("language", "unknown"),
                "genre": meta.get("genre", "その他"),
            }
        except Exception as e:
            last_err = e
            if is_rate_error(e):
                backoff = base_sleep * (2 ** min(attempt, 6))
                backoff = backoff + random.uniform(0, 0.5)
                time.sleep(backoff)
            else:
                if retries == -1:
                    time.sleep(base_sleep)
                    continue
                if attempt >= retries:
                    break
                time.sleep(base_sleep * attempt)
        if retries != -1 and attempt >= retries:
            break

    return {
        "id": rec["clean_id"],
        "original_id": rec["original_id"],
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

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset/base_merged.jsonl")
    ap.add_argument("--output", default="dataset/cleaned.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3, help="-1 で無限リトライ")
    # --- 追加: エラー行の original_id を再処理し、同じ original_id の行を上書き保存 ---
    ap.add_argument("--retry-errors", action="store_true",
                    help="既存の --output を読み、error のある original_id のみ再処理してファイル内のエラー行を上書き保存（id維持）")
    args = ap.parse_args()

    max_workers = min(args.max_workers, 50)

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # 入力読み込み
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
                "original_id": src.get("id"),
                "text": src.get("text") or "",
                "source_kind": src.get("source_kind") or "",
            })
            clean_id += 1
            if args.limit and len(records) >= args.limit:
                break

    # 既存出力から error の original_id を抽出（idは維持）＋ 既存 id マップ構築
    retry_ids = set()
    id_by_oid: Dict[Any, int] = {}    # original_id -> id（既存の id を保持）
    present_oids = set()              # メイン出力に存在する original_id
    max_existing_id = 0

    err_path = outp.with_suffix(outp.suffix + ".errors.jsonl")

    if args.retry_errors and outp.exists():
        # メイン出力の走査（エラー行は今後書かれない想定だが、互換のため両方見る）
        with outp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                oid2 = obj.get("original_id")
                cid2 = obj.get("id")
                if oid2 is not None:
                    present_oids.add(oid2)
                if isinstance(cid2, int) and cid2 > max_existing_id:
                    max_existing_id = cid2
                # 既存の id を確定（先勝ち）
                if oid2 is not None and cid2 is not None and oid2 not in id_by_oid:
                    id_by_oid[oid2] = cid2
                # 互換: メインに残っている error 行があれば収集
                if obj.get("error") and oid2 is not None:
                    retry_ids.add(oid2)

        # エラーファイルがあれば、そちらからも retry 対象を収集
        if err_path.exists():
            with err_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("error"):
                        oid = obj.get("original_id")
                        if oid is not None:
                            retry_ids.add(oid)

        if retry_ids:
            # 入力側を再試行対象に絞る
            records = [r for r in records if r.get("original_id") in retry_ids]
            # 再処理レコードに既存 id を引き継ぐ（id維持）
            for r in records:
                oid = r["original_id"]
                if oid in id_by_oid:
                    r["clean_id"] = id_by_oid[oid]
            missing = len(retry_ids - set(r.get("original_id") for r in records))
            msg = f"retry-errors: {len(retry_ids)} 件検出 -> 入力側の対象 {len(records)} 件にフィルタ（id維持）"
            if missing > 0:
                msg += f" / 入力に見つからなかった id: {missing} 件（--limit などに注意）"
            print(msg)
        else:
            print("retry-errors: 既存出力・エラーファイルに error 行が見つかりませんでした")

    total = len(records)
    if total == 0:
        print("no records.")
        return

    client = get_client()

    done_count = 0
    print_progress(done_count, total)

    lock = threading.Lock()
    new_results: List[Dict[str,Any]] = []

    # retry-errors のときはメモリに結果を溜めて後で置換。通常は追記。
    append_mode = not args.retry_errors
    outf = None
    errf = None
    if append_mode:
        outf = outp.open("a", encoding="utf-8")
        # エラーはメインに書かず、別ファイルにのみ書く（必要な場合の再試行に備える）
        errf = err_path.open("a", encoding="utf-8")
    skipped_errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one, client, rec, args.retries) for rec in records]
        for fut in as_completed(futures):
            result = fut.result()
            with lock:
                if append_mode:
                    # エラーはメイン出力に一切書かない
                    if result.get("error"):
                        if errf:
                            errf.write(json.dumps(result, ensure_ascii=False) + "\n")
                            errf.flush()
                        skipped_errors += 1
                    else:
                        outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                        outf.flush()
                else:
                    new_results.append(result)
            done_count += 1
            print_progress(done_count, total)

    if append_mode:
        if outf: outf.close()
        if errf: errf.close()
        print(f"done -> {outp} (written {total - skipped_errors}, skipped errors {skipped_errors} -> {err_path})")
        return

    # ---- 上書き保存（retry-errors）----
    # 成功した再処理結果のみ置換候補（error を含むものは捨てる）
    new_by_oid: Dict[Any, Dict[str,Any]] = {
        r["original_id"]: r for r in new_results
        if "original_id" in r and not r.get("error")
    }
    tmp_path = outp.with_suffix(outp.suffix + ".tmp")

    replaced = 0
    written_oids = set()

    with outp.open(encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                # 壊れた行はそのまま残す
                fout.write(line)
                continue
            oid = obj.get("original_id")
            # エラー行のみ置換（成功再処理がある場合のみ）。id は旧値を維持
            if obj.get("error") and oid in retry_ids and oid in new_by_oid:
                new_obj = dict(new_by_oid[oid])
                if "id" in obj:
                    new_obj["id"] = obj["id"]  # ★ 旧 id を維持
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                written_oids.add(oid)
                replaced += 1
            else:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # 出力（メイン）に存在しなかった original_id を末尾追記（成功結果のみ）
        # 既存 id があれば維持、なければ最大 id の続きで採番
        for oid, obj in new_by_oid.items():
            if oid not in present_oids and oid not in written_oids:
                new_obj = dict(obj)
                if oid in id_by_oid:
                    new_obj["id"] = id_by_oid[oid]
                else:
                    max_existing_id += 1
                    new_obj["id"] = max_existing_id
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    import os as _os
    _os.replace(tmp_path, outp)

    print(f"replaced {replaced} error rows in {outp}")
    print(f"done -> {outp} ({len(new_results)} retried records; successful {len(new_by_oid)}, failed {len(new_results) - len(new_by_oid)})")

if __name__ == "__main__":
    main()
