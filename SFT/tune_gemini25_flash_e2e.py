#!/usr/bin/env python
"""
ローカル JSONL を GCS にアップロードして、
Gemini 2.5 Flash の教師ありファインチューニング(SFT)ジョブを投げるスクリプト。

1. 設定欄を書き換える
2. python tune_gemini25_flash_e2e.py
"""

import os
import time
from pathlib import Path

from google.cloud import storage
import vertexai
from vertexai.tuning import sft


# ========= 設定ここから =========

# GCP プロジェクト
PROJECT_ID = "your-project-id"

# チューニング対応リージョン（公式サンプルと同じく us-central1 を例に）
# ※必要なら gemini supervised tuning の「Supported regions」ドキュメントに合わせて変更
LOCATION = "us-central1"

# 既存の GCS バケット名
BUCKET_NAME = "your-tuning-bucket"

# GCS 上での保存プレフィックス
GCS_PREFIX = "gemini25_sft/run1"

# ローカルの学習/検証データ
LOCAL_TRAIN_FILE = "data/train_clean.jsonl"   # ←あなたのクリーンデータ
LOCAL_VALID_FILE = ""                         # 検証を使うならパス、不要なら空文字 or None

# ベースモデル（Gemini 2.5 Flash）
SOURCE_MODEL = "gemini-2.5-flash"

# チューニング済みモデルの表示名
TUNED_MODEL_DISPLAY_NAME = "my-gemini25-flash-sft"

# ハイパーパラメータ（全部 None なら「自動」におまかせ）
# いじりたくなったらコメントアウト外す
EPOCHS = None              # 例: 3
ADAPTER_SIZE = None        # 例: 4  (サポート: 1,2,4,8,16 など)
LR_MULTIPLIER = None       # 例: 1.0

POLL_INTERVAL_SEC = 60     # 状態ポーリング間隔（秒）

# ========= 設定ここまで =========


def upload_to_gcs(local_path: str, bucket_name: str, dest_blob: str) -> str | None:
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
    #    認証は gcloud auth application-default login か、
    #    GOOGLE_APPLICATION_CREDENTIALS 環境変数でサービスアカウントキーを指す
    print(f"[INFO] Vertex AI init: project={PROJECT_ID}, location={LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # 3. SFT ジョブ作成（公式サンプルと同じ sft.train を使用）
    print("[INFO] Supervised Fine-Tuning ジョブを作成します...")
    sft_tuning_job = sft.train(
        source_model=SOURCE_MODEL,
        train_dataset=train_uri,
        # 以下はオプション
        validation_dataset=valid_uri,
        tuned_model_display_name=TUNED_MODEL_DISPLAY_NAME,
        epochs=EPOCHS,
        adapter_size=ADAPTER_SIZE,
        learning_rate_multiplier=LR_MULTIPLIER,
    )

    print(f"[INFO] Tuning Job Resource: {sft_tuning_job.resource_name}")
    # 例: projects/xxxx/locations/us-central1/tuningJobs/123456789012345

    # 4. （任意）ジョブ完了までポーリング
    print("[INFO] チューニング完了まで待機します...")
    while not sft_tuning_job.has_ended:
        print(f"    現在の状態: {sft_tuning_job.state}  -> {POLL_INTERVAL_SEC} 秒スリープ")
        time.sleep(POLL_INTERVAL_SEC)
        sft_tuning_job.refresh()

    print(f"[INFO] チューニング完了: state={sft_tuning_job.state}")
    # チューニング済みモデル / エンドポイント名をログ出力
    try:
        print(f"[INFO] tuned_model_name          : {sft_tuning_job.tuned_model_name}")
        print(f"[INFO] tuned_model_endpoint_name: {sft_tuning_job.tuned_model_endpoint_name}")
        print(f"[INFO] experiment                : {sft_tuning_job.experiment}")
    except Exception as e:
        print(f"[WARN] 追加情報の取得でエラー: {e}")


if __name__ == "__main__":
    main()
