#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1 の学習データ (input_text/output_text JSONL) をシードとして、
v2 チューニング済みモデル (voice-v2) で予測変換データを合成するスクリプト。

- 入力: v1 Bison 形式 JSONL
    例: {"input_text": "...", "output_text": "..."}

- 出力:
    - Bison 形式 JSONL:
        input_text / output_text に加え、
        seed_id / seed / context_id / context_prompt / status を含める
    - Gemini SFT 形式 JSONL:
        contents 構造のみ（次の SFT の train.jsonl 用）

- 機能:
    - レート制限系エラーのリトライ（指数バックオフ）
    - 途中再開:
        既存の Bison 出力の (seed_id, context_id) を見てスキップ
    - 固定プロンプト + 4 文脈プロンプト + シードを user 入力に含める
    - Gemini 2.5 Flash 系チューニングモデルを thinking_budget=0（思考オフ）で呼び出し

- 注意:
    - モデル呼び出しに失敗した (seed_id, context_id) の組は
      Bison / Gemini どちらの出力ファイルにも一切書き出さない。
"""

import sys
import json
import time
import random
import threading
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import argparse

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig


# ========= デフォルト設定ここから =========

# Vertex AI プロジェクト & リージョン
PROJECT_ID_DEFAULT = "project-voice-476504"
LOCATION_DEFAULT = "us-central1"

# v2 tuned model の「エンドポイント名」
# projects/{project}/locations/{location}/endpoints/{endpoint_id}
TUNED_MODEL_ENDPOINT_DEFAULT = (
    "projects/700129023625/locations/us-central1/endpoints/8481640393245458432"
)

# v1 の Bison 形式 JSONL (input_text / output_text)
V1_BISON_JSONL_DEFAULT = (
    "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v1/"
    "gemini_sft_yomi2surface_train_mixed_with_context_10k.jsonl"
)

# 出力ファイル
OUT_BISON_JSONL_DEFAULT = (
    "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v2/"
    "voice_v2_synth_from_v1_bison.jsonl"
)
OUT_GEMINI_JSONL_DEFAULT = (
    "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v2/"
    "voice_v2_synth_from_v1_gemini_train.jsonl"
)

# 並列実行数
MAX_WORKERS_DEFAULT = 4

# モデル呼び出しの最大リトライ回数（-1 でレート制限系のみ無限リトライ）
MAX_MODEL_RETRIES_DEFAULT = -1

# thinking_config で thinking_budget=0（思考オフ）を指定
GEN_CONFIG_DEFAULT = GenerateContentConfig(
    temperature=0.2,
    max_output_tokens=64,
    thinking_config=ThinkingConfig(
        thinking_budget=0,  # 0 で思考無効
    ),
)

# v2 学習時の固定プロンプト
FIXED_PROMPT = "キーボードの予測変換として以下に続く言葉を予測してください。"

# 4つの文脈プロンプト
CONTEXTS: List[Tuple[str, str]] = [
    ("dev",      "開発の文脈で予測変換してください。"),
    ("meeting",  "ミーティングの文脈で予測変換してください。"),
    ("casual",   "カジュアルな文脈で予測変換してください。"),
    ("business", "ビジネスの文脈で予測変換してください。"),
]

# 進捗バー幅
PROGRESS_BAR_WIDTH = 50

# ========= デフォルト設定ここまで =========


# --- ユーティリティ ----------------------------------------------------------


def is_rate_error(e: Exception) -> bool:
    """レート制限 / 一時的エラーっぽいかどうかをざっくり判定。"""
    msg = str(e).lower()
    keywords = [
        "429",
        "resource_exhausted",
        "rate",
        "quota",
        "too many requests",
        "temporarily unavailable",
        "deadline exceeded",
        "503",
    ]
    return any(k in msg for k in keywords)


def is_not_found_error(e: Exception) -> bool:
    """404 Not Found 系かざっくり判定。"""
    msg = str(e).lower()
    return ("404" in msg) or ("not found" in msg)


def print_progress(done: int, total: int) -> None:
    """簡易進捗バー表示。"""
    if total <= 0:
        return
    ratio = float(done) / float(total)
    filled = int(PROGRESS_BAR_WIDTH * ratio)
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    percent = int(ratio * 100)
    sys.stdout.write(f"\r[{bar}] {percent:3d}% ({done}/{total})")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def load_processed_pairs(path: Path) -> Set[Tuple[int, str]]:
    """
    既存の Bison 出力ファイルから (seed_id, context_id) のペア集合を復元。
    再実行時にここに含まれる組はスキップして途中再開する。
    """
    processed: Set[Tuple[int, str]] = set()
    if not path.exists():
        return processed

    lines = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            seed_id = obj.get("seed_id")
            ctx_id = obj.get("context_id")
            if isinstance(seed_id, int) and isinstance(ctx_id, str):
                processed.add((seed_id, ctx_id))

    print(
        f"[DEBUG] existing Bison lines={lines}, "
        f"unique processed pairs={len(processed)}",
        file=sys.stderr,
    )
    return processed


def build_full_input(seed_text: str, context_prompt: str) -> str:
    """
    学習時の設定に合わせて user 入力を構成。
    - 文脈プロンプト
    - 固定プロンプト
    - シード（v1 の input_text）
    """
    return f"{context_prompt}\n{FIXED_PROMPT}\n{seed_text}"


# --- モデル呼び出し（リトライ付き, google-genai） ---------------------------


def call_model(
    client: genai.Client,
    model_endpoint: str,
    full_input: str,
    retries: int,
    base_sleep: float = 0.5,
) -> str:
    """
    google-genai Client でテキスト生成。
    - retries == -1 でレート制限系のみ無限リトライ
    - 404 / その他の恒久エラーは retries==-1 でも打ち切る
    """
    attempt = 0
    last_err: Optional[Exception] = None

    while True:
        attempt += 1
        try:
            resp = client.models.generate_content(
                model=model_endpoint,
                contents=full_input,
                config=GEN_CONFIG_DEFAULT,
            )
            text = (resp.text or "").strip()
            if not text:
                raise RuntimeError("Model returned empty text.")
            return text

        except Exception as e:  # noqa: BLE001
            last_err = e
            rate = is_rate_error(e)
            not_found = is_not_found_error(e)
            print(
                f"[WARN] model call failed (attempt={attempt}, "
                f"is_rate={rate}, not_found={not_found}): {e}",
                file=sys.stderr,
            )

            # 404 / Not Found は恒久エラーとして即終了
            if not_found:
                break

            # レート制限・一時エラー → バックオフ（retries==-1 なら無限リトライ）
            if rate:
                backoff = base_sleep * (2 ** min(attempt, 6))
                backoff += random.uniform(0, 0.5)
                time.sleep(backoff)
                continue

            # その他のエラー
            if retries == -1:
                # 無限リトライ指定でも、非レートエラーは無限には回さない
                break

            if attempt >= retries:
                break

            time.sleep(base_sleep * attempt)

    if last_err is None:
        last_err = RuntimeError("Unknown error in call_model.")
    raise last_err


def process_one(
    client: genai.Client,
    model_endpoint: str,
    rec: Dict[str, Any],
    retries: int,
) -> Dict[str, Any]:
    """
    1 (seed_id, seed, context) の組を処理。

    成功時: status="ok" でフル情報を返す。
    失敗時: status="error" のみ返し、出力ファイルには一切書き込まない（呼び出し側でスキップ）。
    """
    seed_id = rec["seed_id"]
    seed = rec["seed"]
    ctx_id = rec["context_id"]
    ctx_prompt = rec["context_prompt"]

    full_input = build_full_input(seed, ctx_prompt)

    try:
        out_text = call_model(client, model_endpoint, full_input, retries=retries)
        status = "ok"
    except Exception as e:  # noqa: BLE001
        print(
            f"[ERROR] giving up on seed_id={seed_id}, ctx={ctx_id}: {e}",
            file=sys.stderr,
        )
        # エラー時は seed も含め、一切書き出さない方針なので minimum 情報のみ返す
        return {
            "status": "error",
            "seed_id": seed_id,
            "context_id": ctx_id,
        }

    return {
        "seed_id": seed_id,
        "seed": seed,
        "context_id": ctx_id,
        "context_prompt": ctx_prompt,
        "input_text": full_input,
        "output_text": out_text,
        "status": status,
    }


# --- メイン -------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="v1 学習データをシードにして、v2 tuned model で予測変換データを合成するスクリプト"
    )
    ap.add_argument("--project-id", default=PROJECT_ID_DEFAULT)
    ap.add_argument("--location", default=LOCATION_DEFAULT)
    ap.add_argument(
        "--model-endpoint",
        default=TUNED_MODEL_ENDPOINT_DEFAULT,
        help=(
            "tuned model のエンドポイント名 "
            "(例: projects/PROJECT/locations/REGION/endpoints/ENDPOINT_ID)"
        ),
    )
    ap.add_argument("--v1-bison-jsonl", default=V1_BISON_JSONL_DEFAULT)
    ap.add_argument("--out-bison", default=OUT_BISON_JSONL_DEFAULT)
    ap.add_argument("--out-gemini", default=OUT_GEMINI_JSONL_DEFAULT)
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS_DEFAULT)
    ap.add_argument(
        "--retries",
        type=int,
        default=MAX_MODEL_RETRIES_DEFAULT,
        help="-1 でレート制限系のみ無限リトライ",
    )

    args = ap.parse_args()

    project_id = args.project_id
    location = args.location
    model_endpoint = args.model_endpoint
    v1_path = Path(args.v1_bison_jsonl)
    out_bison_path = Path(args.out_bison)
    out_gemini_path = Path(args.out_gemini)
    max_workers = max(1, min(args.max_workers, 50))
    retries = args.retries

    if not v1_path.exists():
        print(f"[ERROR] v1 Bison JSONL not found: {v1_path}", file=sys.stderr)
        sys.exit(1)

    out_bison_path.parent.mkdir(parents=True, exist_ok=True)
    out_gemini_path.parent.mkdir(parents=True, exist_ok=True)

    # 既存 Bison 出力から (seed_id, context_id) セットを復元して、途中再開を実現
    processed_pairs = load_processed_pairs(out_bison_path)

    # v1 JSONL 読み込み: 各行の input_text を seed として扱う
    seeds: List[str] = []
    with v1_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seed_text = obj.get("input_text", "")
            if not isinstance(seed_text, str) or not seed_text:
                continue
            seeds.append(seed_text)

    if not seeds:
        print(f"[ERROR] no seeds found in {v1_path}", file=sys.stderr)
        sys.exit(1)

    # これから処理する (seed_id, seed, context) 組を作る
    records: List[Dict[str, Any]] = []
    for seed_id, seed in enumerate(seeds):
        for ctx_id, ctx_prompt in CONTEXTS:
            key = (seed_id, ctx_id)
            if key in processed_pairs:
                continue
            records.append(
                {
                    "seed_id": seed_id,
                    "seed": seed,
                    "context_id": ctx_id,
                    "context_prompt": ctx_prompt,
                }
            )

    total = len(records)
    print(
        f"[INFO] v1 seeds={len(seeds)}, "
        f"new seed-context pairs to process={total}",
        file=sys.stderr,
    )

    if total == 0:
        print("[INFO] nothing to do. all pairs already processed.", file=sys.stderr)
        return

    # Google Gen AI SDK クライアント（Vertex AI モード）初期化
    print(
        f"[INFO] genai.Client(vertexai=True, project={project_id}, location={location})",
        file=sys.stderr,
    )
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    print(f"[INFO] using tuned model endpoint: {model_endpoint}", file=sys.stderr)

    # 出力ファイルは追記モード
    f_bison = out_bison_path.open("a", encoding="utf-8")
    f_gemini = out_gemini_path.open("a", encoding="utf-8")
    lock = threading.Lock()

    done_count = 0
    print_progress(done_count, total)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_one, client, model_endpoint, rec, retries)
            for rec in records
        ]

        print(
            f"[DEBUG] submitted {len(futures)} tasks with max_workers={max_workers}",
            file=sys.stderr,
        )

        for fut in as_completed(futures):
            result = fut.result()

            # 失敗したレコードは出力ファイルに書かない
            if result.get("status") != "ok":
                done_count += 1
                print_progress(done_count, total)
                continue

            # Bison 形式
            bison_line = {
                "seed_id": result["seed_id"],
                "seed": result["seed"],
                "context_id": result["context_id"],
                "context_prompt": result["context_prompt"],
                "input_text": result["input_text"],
                "output_text": result["output_text"],
                "status": result["status"],
            }

            # Gemini SFT 形式
            gemini_line = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": result["input_text"]}],
                    },
                    {
                        "role": "model",
                        "parts": [{"text": result["output_text"]}],
                    },
                ]
            }

            with lock:
                f_bison.write(json.dumps(bison_line, ensure_ascii=False) + "\n")
                f_bison.flush()
                f_gemini.write(json.dumps(gemini_line, ensure_ascii=False) + "\n")
                f_gemini.flush()

            done_count += 1
            if done_count <= 5 or done_count % 100 == 0:
                print(
                    f"[DEBUG] progress {done_count}/{total}",
                    file=sys.stderr,
                )
            print_progress(done_count, total)

    f_bison.close()
    f_gemini.close()

    print(
        f"[INFO] done ->\n  Bison : {out_bison_path}\n  Gemini: {out_gemini_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
