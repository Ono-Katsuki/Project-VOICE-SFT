## 使い方

1. 上のスクリプトを `tune_gemini25_flash_e2e.py` として保存。
2. スクリプト先頭の **設定ブロック** をあなたの環境に合わせて編集：

   * `PROJECT_ID`
   * `LOCATION`（チューニング対応リージョン）
   * `BUCKET_NAME`
   * `GCS_PREFIX`（任意のパスでOK）
   * `LOCAL_TRAIN_FILE` / `LOCAL_VALID_FILE`
   * `TUNED_MODEL_DISPLAY_NAME`
3. 認証（まだなら）：

   ```bash
   gcloud auth application-default login
   ```
4. 実行：

   ```bash
   python tune_gemini25_flash_e2e.py
   ```

