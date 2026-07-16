# SFT — Gemini 2.5 Flash tuning launcher

Canonical Vertex AI SFT launcher for the Project VOICE predictive-text
models. `tune_gemini25_flash_e2e.py` is the reference entry point;
`main-sequence/vN/train.py` and `post-paper/vN/train.py` are frozen
iteration snapshots.

## Usage

1. Edit the **configuration block** at the top of `tune_gemini25_flash_e2e.py`
   for your environment:

   * `PROJECT_ID`
   * `LOCATION` — a tuning-supported region
   * `BUCKET_NAME`
   * `GCS_PREFIX` — any path prefix
   * `LOCAL_TRAIN_FILE` / `LOCAL_VALID_FILE`
   * `TUNED_MODEL_DISPLAY_NAME`

2. Authenticate (if you have not already):

   ```bash
   gcloud auth application-default login
   ```

3. Run:

   ```bash
   python tune_gemini25_flash_e2e.py
   ```

`tune_gemini25_flash_e2e.py` submits the SFT job directly. The v11
snapshot (`main-sequence/v11/train.py`) additionally estimates cost up
front and refuses to submit above `MAX_BUDGET_JPY` (default ¥50,000);
override that guard via the constants at the top of `v11/train.py` or
via the `USDJPY` / `SFT_COST_SAFETY_MULTIPLIER` environment variables.
