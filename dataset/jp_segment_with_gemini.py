#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import sys
import time
import threading
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from google import genai
except ImportError:
    print("pip install google-genai が必要です", file=sys.stderr)
    sys.exit(1)

# --- モデル・設定 -----------------------------------------------------------

MODEL_SEG = "gemini-2.5-flash"
NO_THINKING_CONFIG = {"thinking_config": {"thinking_budget": 0}}

# 評価NGのときに、プロンプトにフィードバックを足して再トライする最大回数
MAX_EVAL_RETRIES = 2

SEGMENT_PROMPT = """あなたは日本語テキストを形態素単位に分割し、
各形態素のひらがな読みも付与するアシスタントです。

【タスク】
入力されたテキストを、以下の2つの文字列に変換してください。

1. surface: 原文を形態素ごとに「/」で区切った文字列
2. yomi: 1. と同じ区切りで、各形態素をひらがなにした文字列

【例】
入力:
今日は天気が良いです。

出力:
surface: 今日/は/天気/が/良い/です/。
yomi: きょう/は/てんき/が/いい/です/。

【ルール】
- 記号（。、「」、？！など）もひとつのトークンとして扱い、
  対応する位置に「/」を入れてください。
- カタカナ語は原則ひらがなに変換してください（「テスト」→「てすと」）。
- 英数字やURLなど、ひらがなにしづらい部分はそのままで構いません。
- 入力が日本語でない場合は、分割せずにそのまま surface として出し、
  yomi も surface と同じ文字列を出してください。

【厳守する条件】
- surface 内のすべての「/」を取り除いた文字列は、
  入力テキストと完全に一致させてください。
  （空白・改行・記号も含めて、一文字も変更しないこと）
- surface を「/」で分割したトークン列と、
  yomi を「/」で分割したトークン列の長さは必ず同じにしてください。
- yomi の各トークンは、対応する surface のトークンの読みであるようにしてください。
- 出力は UTF-8 で扱われる前提です。制御文字などは含めないでください。

【出力フォーマット】
- 次の JSON を <SEG> と </SEG> の間に1行だけ出力してください。

<SEG>
{"surface": "今日/は/天気/が/良い/です/。", "yomi": "きょう/は/てんき/が/いい/です/。"}
</SEG>

- <SEG> タグの外側には何も出力しないでください。
"""


# --- 共通ユーティリティ -----------------------------------------------------

def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("環境変数 GOOGLE_API_KEY もしくは GEMINI_API_KEY がありません", file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=api_key)


def is_rate_error(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "rate" in msg.lower() or "503" in msg


# --- モデル出力のパース＆評価＆補正 --------------------------------------------

def _parse_segment_response(full: str) -> Tuple[bool, str, str, str]:
    """
    モデルの生出力 full から surface / yomi を取り出す。
    戻り値: (ok, surface, yomi, reason_if_ng)
    """
    m = re.search(r"<SEG>\s*(\{.*?\})\s*</SEG>", full, re.DOTALL | re.IGNORECASE)
    raw = m.group(1).strip() if m else full

    try:
        obj = json.loads(raw)
        surface = str(obj.get("surface", ""))
        yomi = str(obj.get("yomi", ""))
        if not surface:
            return False, "", "", "surface が空です。"
        if not yomi:
            return False, "", "", "yomi が空です。"
        return True, surface, yomi, ""
    except Exception as e:
        return False, "", "", f"モデルの出力を JSON として解釈できませんでした: {e}"


def _normalize_for_compare(s: str) -> str:
    """
    原文一致チェック用の軽い正規化。
    必要に応じて CRLF→LF などをここに追加してもよい。
    今回は「完全一致」を狙うのでそのまま返す。
    """
    return s


def evaluate_segment(original_text: str, surface: str, yomi: str) -> Tuple[bool, str]:
    """
    surface / yomi が仕様を満たしているかチェックする。
    戻り値: (ok, reason_if_ng)
    """
    if not isinstance(surface, str) or not isinstance(yomi, str):
        return False, "surface または yomi が文字列ではありません。"

    s_tokens = surface.split("/")
    y_tokens = yomi.split("/")

    if len(s_tokens) != len(y_tokens):
        return False, (
            f"surface のトークン数 ({len(s_tokens)}) と "
            f"yomi のトークン数 ({len(y_tokens)}) が一致していません。"
        )

    # 原文一致チェック（トークンを連結）
    reconstructed = "".join(s_tokens)
    if _normalize_for_compare(reconstructed) != _normalize_for_compare(original_text):
        return False, (
            "surface を \"/\" で分割して連結した文字列が、入力テキストと一致していません。\n"
            f"  reconstructed: {reconstructed[:80]!r}\n"
            f"  original     : {original_text[:80]!r}"
        )

    return True, ""


def try_repair_segment(
    original_text: str,
    surface: str,
    yomi: str,
) -> Tuple[bool, str, str, str]:
    """
    プログラム側で直せる範囲の補正を試みる。

    戻り値:
        (ok, fixed_surface, fixed_yomi, reason_if_ng)
    """
    # まずは as-is で評価
    ok, reason = evaluate_segment(original_text, surface, yomi)
    if ok:
        return True, surface, yomi, ""

    # --- 1. 空トークン対処: "A//B" みたいに "" が入っているケース ---
    s_tokens = surface.split("/")
    y_tokens = yomi.split("/")

    if len(s_tokens) != len(y_tokens):
        # トークン数自体が違うなら、ここでは触らない（LLM リトライに任せる）
        return False, surface, yomi, reason

    new_s = []
    new_y = []
    removed_empty = False
    for s_tok, y_tok in zip(s_tokens, y_tokens):
        if s_tok == "" and y_tok == "":
            removed_empty = True
            continue
        new_s.append(s_tok)
        new_y.append(y_tok)

    if removed_empty:
        surface = "/".join(new_s)
        yomi = "/".join(new_y)
        s_tokens = new_s
        y_tokens = new_y

    # 再評価
    ok2, reason2 = evaluate_segment(original_text, surface, yomi)
    if ok2:
        return True, surface, yomi, ""

    # --- 2. prefix/suffix 補完: 原文の部分文字列になっているだけのパターン ---
    rec = "".join(s_tokens)
    orig = original_text

    if rec and rec in orig:
        idx = orig.find(rec)
        prefix = orig[:idx]
        suffix = orig[idx + len(rec):]

        changed = False
        if prefix:
            s_tokens.insert(0, prefix)
            y_tokens.insert(0, prefix)
            changed = True
        if suffix:
            s_tokens.append(suffix)
            y_tokens.append(suffix)
            changed = True

        if changed:
            surface_fixed = "/".join(s_tokens)
            yomi_fixed = "/".join(y_tokens)
            ok3, reason3 = evaluate_segment(orig, surface_fixed, yomi_fixed)
            if ok3:
                return True, surface_fixed, yomi_fixed, ""
            else:
                return False, surface_fixed, yomi_fixed, reason3

    # ここまで来たら自動補正では無理
    return False, surface, yomi, reason


def call_segment(client: genai.Client, text: str) -> Dict[str, str]:
    """
    Gemini に分割＆よみをやらせる。
    - JSON フォーマット
    - surface/yomi トークン数一致
    - surface の連結で原文一致
    をチェックし、
    まずプログラム側で自動補正を試す。
    それでも NG の場合だけプロンプトにフィードバックを足して再トライする。
    """
    feedback = ""
    last_full_output = ""

    for attempt in range(MAX_EVAL_RETRIES + 1):
        # ベースプロンプト + これまでの評価フィードバック
        instr = SEGMENT_PROMPT
        if feedback:
            instr += (
                "\n\n【前回出力に対する追加指示】\n"
                "前回の出力は、指定した評価条件を満たしていませんでした。\n"
                "以下のフィードバックを必ず反映させて、条件をすべて満たすように修正してください。\n"
                f"{feedback}\n"
            )

        resp = client.models.generate_content(
            model=MODEL_SEG,
            contents=[instr, text],
            config=NO_THINKING_CONFIG,
        )
        full = (resp.text or "").strip()
        last_full_output = full

        ok_parse, surface, yomi, reason_parse = _parse_segment_response(full)
        if not ok_parse:
            feedback = (
                "出力が指定された JSON 形式になっていません。\n"
                "必ず <SEG> と </SEG> の間に、"
                '{"surface": "...", "yomi": "..."}'
                " という1つの JSON オブジェクトだけを1行で出力してください。\n"
                f"エラーの詳細: {reason_parse}"
            )
            print(f"[DEBUG] parse NG attempt={attempt+1}: {reason_parse}", file=sys.stderr)
            continue

        # ★ ここでまず自動補正を試す
        ok_fixed, surface_fixed, yomi_fixed, reason_eval = try_repair_segment(
            text, surface, yomi
        )
        if ok_fixed:
            print(f"[DEBUG] segment OK attempt={attempt+1}", file=sys.stderr)
            return {"surface": surface_fixed, "yomi": yomi_fixed}

        # 自動補正でもダメだった場合だけ LLM にフィードバック
        feedback = (
            "出力が次の評価条件を満たしていませんでした。\n"
            f"{reason_eval}\n\n"
            "上記の問題をすべて修正し、"
            "入力テキストの文字列を一切変更せずに surface を構成し直してください。"
        )
        print(f"[DEBUG] evaluate NG attempt={attempt+1}: {reason_eval}", file=sys.stderr)

    # ここまで来たらすべての試行で NG → フォールバック
    print("[ERROR] call_segment: 評価付き再トライに失敗したためフォールバックします。", file=sys.stderr)
    print(f"[ERROR] last model output: {last_full_output[:500]}", file=sys.stderr)

    return {"surface": text, "yomi": text}


# --- 1レコード処理 ------------------------------------------------------------

def process_one(
    shared_client: genai.Client,
    rec: Dict[str, Any],
    retries: int = 3,
    base_sleep: float = 0.5,
) -> Dict[str, Any]:
    """
    1レコードを処理する。
    retries == -1 の場合は無限リトライ。
    """
    attempt = 0
    last_err = None
    orig_id = rec.get("original_id")

    print(f"[DEBUG] process_one START original_id={orig_id}", file=sys.stderr)

    while True:
        attempt += 1
        try:
            seg = call_segment(shared_client, rec["text"])

            print(
                f"[DEBUG] process_one SUCCESS original_id={orig_id} attempt={attempt}",
                file=sys.stderr,
            )

            out = {
                "id": rec.get("id"),
                "original_id": orig_id,
                "text": rec["text"],
                "surface": seg["surface"],
                "yomi": seg["yomi"],
            }
            # clean_score など他のフィールドはそのまま引き継ぐ
            for k, v in rec.items():
                if k not in out:
                    out[k] = v
            return out

        except Exception as e:
            last_err = e
            is_rate = is_rate_error(e)
            print(
                f"[DEBUG] process_one EXCEPTION original_id={orig_id} "
                f"attempt={attempt} is_rate_error={is_rate} error={e}",
                file=sys.stderr,
            )

            if is_rate:
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

    if last_err is not None:
        print(
            f"[ERROR] segment failed for original_id={orig_id}: {last_err}",
            file=sys.stderr,
        )

    print(f"[DEBUG] process_one FALLBACK original_id={orig_id}", file=sys.stderr)

    out = {
        "id": rec.get("id"),
        "original_id": orig_id,
        "text": rec["text"],
        "surface": rec["text"],
        "yomi": rec["text"],
    }
    for k, v in rec.items():
        if k not in out:
            out[k] = v
    return out


# --- 進捗表示 -----------------------------------------------------------------

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


# --- メイン -------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset/cleaned.jsonl")
    ap.add_argument("--output", default="dataset/segmented.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3, help="-1 で無限リトライ")
    args = ap.parse_args()

    max_workers = min(args.max_workers, 50)

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # 既存 segmented.jsonl から処理済み original_id を読む（途中再開用）
    processed_original_ids = set()
    existing_lines = 0

    if outp.exists():
        with outp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                existing_lines += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                oid = obj.get("original_id")
                if oid is not None:
                    processed_original_ids.add(oid)

    print(
        f"[DEBUG] existing segmented.jsonl lines={existing_lines} "
        f"unique_processed_original_ids={len(processed_original_ids)}",
        file=sys.stderr,
    )

    records: List[Dict[str, Any]] = []
    total_input_lines = 0

    with inp.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_input_lines += 1
            src = json.loads(line)
            original_id = src.get("original_id") or src.get("id")

            if original_id in processed_original_ids:
                continue

            rec = dict(src)
            rec["original_id"] = original_id
            rec.setdefault("id", original_id)
            rec.setdefault("text", "")
            records.append(rec)

            if args.limit and len(records) >= args.limit:
                break

    total = len(records)
    print(
        f"[DEBUG] input total_lines={total_input_lines} "
        f"new_records_to_process={total}",
        file=sys.stderr,
    )

    if total == 0:
        print("no new records.")
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

        print(
            f"[DEBUG] submitted {len(futures)} futures with max_workers={max_workers}",
            file=sys.stderr,
        )

        for fut in as_completed(futures):
            result = fut.result()
            with lock:
                outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                outf.flush()
            done_count += 1

            if done_count <= 5 or done_count % 1000 == 0:
                print(
                    f"[DEBUG] progress done_count={done_count}/{total}",
                    file=sys.stderr,
                )

            print_progress(done_count, total)

    outf.close()
    print(f"done -> {outp} ({total} records)")


if __name__ == "__main__":
    main()
