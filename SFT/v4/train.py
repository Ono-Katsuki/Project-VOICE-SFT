
#!/usr/bin/env python
"""
ローカル JSONL を GCS にアップロードして、
Gemini 2.5 Flash の教師ありファインチューニング(SFT)ジョブを投げるスクリプト。
"""

import os
import time
from pathlib import Path
from typing import Optional  # ★ 追加

from google.cloud import storage
import vertexai
from vertexai.tuning import sft


# ========= 設定ここから =========

PROJECT_ID = "project-voice-476504"
LOCATION = "us-central1"
BUCKET_NAME = "project-voice-models"
GCS_PREFIX = "gemini25_sft/run4"

LOCAL_TRAIN_FILE = "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v4/gemini_sft_yomi2surface_train_mixed_with_context_10k_with_prompt_gemini_format.jsonl"
LOCAL_VALID_FILE = ""

SOURCE_MODEL = "gemini-2.5-flash"
TUNED_MODEL_DISPLAY_NAME = "voice-v4"

EPOCHS = None
ADAPTER_SIZE = None
LR_MULTIPLIER = None

POLL_INTERVAL_SEC = 60

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


def main() -> None:
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

    # 2. Vertex AI 初期化
    print(f"[INFO] Vertex AI init: project={PROJECT_ID}, location={LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # 3. SFT ジョブ作成
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

    # 4. ジョブ完了まで待機
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
