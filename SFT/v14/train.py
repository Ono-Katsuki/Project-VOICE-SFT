#!/usr/bin/env python
"""
ローカル JSONL を GCS にアップロードして、
Gemini 2.5 Flash の教師ありファインチューニング(SFT)ジョブを投げるスクリプト。

追加: 実行前にコスト見積もりを行い、5万円以上ならブロックする。
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Any, Dict, List

from google.cloud import storage
import vertexai
from vertexai.tuning import sft

# ===== 追加: token count 用 =====
try:
    # 公式リファレンス: vertexai.generative_models.GenerativeModel.count_tokens
    from vertexai.generative_models import GenerativeModel
except Exception:
    # 互換性のため
    from vertexai.preview.generative_models import GenerativeModel


# ========= 設定ここから =========

PROJECT_ID = "project-voice-476504"
LOCATION = "us-central1"
BUCKET_NAME = "project-voice-models"
GCS_PREFIX = "gemini25_sft/run14"

LOCAL_TRAIN_FILE = "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v14/joint_sent4_words8_from_tone8__strongtone_gemini.words8_wordsonly.top10k.cleaned_refilled8_from_synthv11_tokens_strict.jsonl"
LOCAL_VALID_FILE = ""

SOURCE_MODEL = "gemini-2.5-flash"
TUNED_MODEL_DISPLAY_NAME = "voice-v14"

EPOCHS = 1
ADAPTER_SIZE = None
LR_MULTIPLIER = None

POLL_INTERVAL_SEC = 60

# ===== 追加: 予算ガード設定 =====
MAX_BUDGET_JPY = 50_000

# 為替は近似（必要なら環境変数で上書き）
USDJPY = float(os.getenv("USDJPY", "160"))

# 見積もりの安全係数（控えめに盛る）
SFT_COST_SAFETY_MULTIPLIER = float(os.getenv("SFT_COST_SAFETY_MULTIPLIER", "1.2"))

# トークン比率推定のために実計測するサンプル行数（多いほど精度↑、API呼び出し↑）
TOKEN_COUNT_SAMPLE_LINES = int(os.getenv("TOKEN_COUNT_SAMPLE_LINES", "200"))

# Vertex AI Pricing の SFT 単価（USD / 1M training tokens）
# https://cloud.google.com/vertex-ai/generative-ai/pricing
# ※モデルや料金改定があればここを更新
SFT_PRICE_USD_PER_1M_TRAINING_TOKENS = {
    "gemini-2.5-pro": 25.0,
    "gemini-2.5-flash": 5.0,
    "gemini-2.5-flash-lite": 1.5,
}

# ========= 設定ここまで =========


def upload_to_gcs(local_path: str, bucket_name: str, dest_blob: str) -> Optional[str]:
    """ローカルファイルを GCS にアップロードして gs:// URI を返す。"""
    if not local_path:
        return None

    p = Path(local_path)
    if not p.exists():
        print(f"[WARN] ローカルファイルが見つかりません: {local_path} -> スキップ")
        return None

    print(f"[INFO] GCS へアップロード: {local_path} -> gs://{bucket_name}/{dest_blob}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(local_path)

    uri = f"gs://{bucket_name}/{dest_blob}"
    print(f"[INFO] アップロード完了: {uri}")
    return uri


def _extract_text_chars_from_contents(contents: Any) -> int:
    """
    contents からテキスト部分の文字数を概算。
    形式:
      - list[{"role": "...", "parts":[{"text":"..."}]}]
      - などに対応
    """
    if contents is None:
        return 0
    if isinstance(contents, str):
        return len(contents)

    total = 0
    if isinstance(contents, list):
        for c in contents:
            if isinstance(c, dict):
                parts = c.get("parts", [])
                if isinstance(parts, list):
                    for p in parts:
                        if isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                            total += len(p["text"])
                # parts が無い形式の保険
                if "text" in c and isinstance(c["text"], str):
                    total += len(c["text"])
            elif isinstance(c, str):
                total += len(c)
    elif isinstance(contents, dict):
        # 単発 dict の保険
        if "parts" in contents and isinstance(contents["parts"], list):
            for p in contents["parts"]:
                if isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                    total += len(p["text"])
        if "text" in contents and isinstance(contents["text"], str):
            total += len(contents["text"])

    return total


def _extract_contents_from_jsonl_obj(obj: Dict[str, Any]) -> Any:
    """
    SFT JSONL の1行(obj)から contents を取り出す。
    代表例:
      {"contents":[{"role":"user","parts":[{"text":"..."}]}, {"role":"model","parts":[{"text":"..."}]}]}
    """
    if "contents" in obj:
        return obj["contents"]
    if "messages" in obj:
        return obj["messages"]

    # よくある prompt/completion 形式の保険
    if "prompt" in obj and "completion" in obj:
        return [
            {"role": "user", "parts": [{"text": str(obj["prompt"])}]},
            {"role": "model", "parts": [{"text": str(obj["completion"])}]},
        ]
    if "input" in obj and "output" in obj:
        return [
            {"role": "user", "parts": [{"text": str(obj["input"])}]},
            {"role": "model", "parts": [{"text": str(obj["output"])}]},
        ]

    # 最後の保険: 行全体を文字列化
    return json.dumps(obj, ensure_ascii=False)


def _get_sft_unit_price_usd_per_1m(source_model: str) -> float:
    m = source_model.lower()
    # flash-lite は flash に部分一致するので先に判定
    if "flash-lite" in m:
        return SFT_PRICE_USD_PER_1M_TRAINING_TOKENS["gemini-2.5-flash-lite"]
    if "2.5-pro" in m or "2.5-pro" in m:
        return SFT_PRICE_USD_PER_1M_TRAINING_TOKENS["gemini-2.5-pro"]
    if "2.5-flash" in m:
        return SFT_PRICE_USD_PER_1M_TRAINING_TOKENS["gemini-2.5-flash"]
    raise ValueError(f"未知の SOURCE_MODEL です。単価辞書に追加してください: {source_model}")


def estimate_training_tokens_from_jsonl(
    local_train_file: str,
    source_model: str,
    sample_lines: int = 200,
) -> int:
    """
    JSONL を読み、テキスト文字数を全量集計しつつ、
    先頭 sample_lines 行は count_tokens で実トークンを測って「tokens/char」を推定。
    そこから全体のトークン数を推定して返す。
    """
    p = Path(local_train_file)
    if not p.exists():
        raise FileNotFoundError(f"学習ファイルが見つかりません: {local_train_file}")

    model = GenerativeModel(source_model)

    total_chars = 0
    sample_chars = 0
    sample_tokens = 0
    line_count = 0

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 壊れ行は保険で文字数だけ加算
                total_chars += len(line)
                continue

            contents = _extract_contents_from_jsonl_obj(obj)
            cchars = _extract_text_chars_from_contents(contents)
            total_chars += cchars

            if sample_lines > 0 and line_count <= sample_lines:
                try:
                    resp = model.count_tokens(contents)
                    # CountTokensResponse.total_tokens を想定
                    stoks = int(getattr(resp, "total_tokens", 0))
                    if stoks > 0 and cchars > 0:
                        sample_tokens += stoks
                        sample_chars += cchars
                except Exception as e:
                    # サンプル計測に失敗しても、後段で保守的推定にフォールバック
                    print(f"[WARN] count_tokens 失敗 (line={line_count}): {e}")

    if line_count == 0:
        return 0

    if sample_tokens > 0 and sample_chars > 0:
        tokens_per_char = sample_tokens / sample_chars
        est_tokens = int(total_chars * tokens_per_char)
        print(f"[INFO] token推定: sample {min(line_count, sample_lines)}行で {tokens_per_char:.4f} tokens/char を推定")
    else:
        # フォールバック: 超保守的に「1文字=1トークン」扱い
        est_tokens = int(total_chars)
        print("[WARN] token推定: サンプル計測が取れなかったため 1文字=1トークンで保守推定します")

    print(f"[INFO] 学習データ: {line_count} 行, 文字数合計(概算)={total_chars:,}, 推定tokens={est_tokens:,}")
    return est_tokens


def guard_budget_or_exit(
    local_train_file: str,
    source_model: str,
    epochs: int,
    max_budget_jpy: int,
) -> None:
    """見積もりを出して、max_budget_jpy 以上なら SystemExit で止める。"""
    unit_price = _get_sft_unit_price_usd_per_1m(source_model)

    est_tokens_per_epoch = estimate_training_tokens_from_jsonl(
        local_train_file=local_train_file,
        source_model=source_model,
        sample_lines=TOKEN_COUNT_SAMPLE_LINES,
    )

    training_tokens = est_tokens_per_epoch * int(epochs)
    est_cost_usd = (training_tokens / 1_000_000.0) * unit_price
    est_cost_jpy = est_cost_usd * USDJPY

    # 安全係数で少し盛って判定
    est_cost_jpy_guard = est_cost_jpy * SFT_COST_SAFETY_MULTIPLIER

    print("[INFO] ===== 見積もり =====")
    print(f"[INFO] source_model                  : {source_model}")
    print(f"[INFO] epochs                       : {epochs}")
    print(f"[INFO] unit price (USD/1M tokens)    : {unit_price}")
    print(f"[INFO] USDJPY                        : {USDJPY}")
    print(f"[INFO] training tokens (est)          : {training_tokens:,}")
    print(f"[INFO] est cost (USD)                 : {est_cost_usd:,.4f}")
    print(f"[INFO] est cost (JPY)                 : {est_cost_jpy:,.0f}")
    print(f"[INFO] safety multiplier              : {SFT_COST_SAFETY_MULTIPLIER}")
    print(f"[INFO] est cost w/ safety (JPY)       : {est_cost_jpy_guard:,.0f}")
    print(f"[INFO] budget limit (JPY)             : {max_budget_jpy:,}")
    print("[INFO] ===================")

    if est_cost_jpy_guard >= max_budget_jpy:
        print(
            f"[ABORT] 見積もりが予算上限に到達: {est_cost_jpy_guard:,.0f}円 >= {max_budget_jpy:,}円\n"
            "        実行をブロックしました。"
        )
        raise SystemExit(2)


def main() -> None:
    # 0. Vertex AI 初期化（token count でも利用）
    print(f"[INFO] Vertex AI init: project={PROJECT_ID}, location={LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # 追加: 予算ガード（ここで止まればアップロードもジョブ投入もしない）
    guard_budget_or_exit(
        local_train_file=LOCAL_TRAIN_FILE,
        source_model=SOURCE_MODEL,
        epochs=EPOCHS,
        max_budget_jpy=MAX_BUDGET_JPY,
    )

    # 1. データを GCS にアップロード
    train_uri = upload_to_gcs(
        LOCAL_TRAIN_FILE,
        BUCKET_NAME,
        f"{GCS_PREFIX}/train.jsonl",
    )
    if train_uri is None:
        raise RuntimeError("トレーニングデータが無いので終了します")

    valid_uri = upload_to_gcs(
        LOCAL_VALID_FILE,
        BUCKET_NAME,
        f"{GCS_PREFIX}/valid.jsonl",
    ) if LOCAL_VALID_FILE else None

    # 2. SFT ジョブ作成
    print("[INFO] Supervised Fine-Tuning ジョブを作成します...")
    sft_tuning_job = sft.train(
        source_model=SOURCE_MODEL,
        train_dataset=train_uri,
        validation_dataset=valid_uri,
        tuned_model_display_name=TUNED_MODEL_DISPLAY_NAME,
        epochs=EPOCHS,
        adapter_size=ADAPTER_SIZE,
        learning_rate_multiplier=LR_MULTIPLIER,
    )

    print(f"[INFO] Tuning Job Resource: {sft_tuning_job.resource_name}")

    # 3. ジョブ完了まで待機
    print("[INFO] チューニング完了まで待機します...")
    while not sft_tuning_job.has_ended:
        print(f"    現在の状態: {sft_tuning_job.state}  -> {POLL_INTERVAL_SEC} 秒スリープ")
        time.sleep(POLL_INTERVAL_SEC)
        sft_tuning_job.refresh()

    print(f"[INFO] チューニング完了: state={sft_tuning_job.state}")
    try:
        print(f"[INFO] tuned_model_name          : {sft_tuning_job.tuned_model_name}")
        print(f"[INFO] tuned_model_endpoint_name: {sft_tuning_job.tuned_model_endpoint_name}")
        print(f"[INFO] experiment                : {sft_tuning_job.experiment}")
    except Exception as e:
        print(f"[WARN] 追加情報の取得でエラー: {e}")


if __name__ == "__main__":
    main()
