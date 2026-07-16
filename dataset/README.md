# `dataset/` — data pipeline for the Gemini SFT models

Most scripts here call **base Gemini 2.5 Flash** through the API as a
text-processing utility (cleaning, segmentation, historical synthesis
attempts). A few — listed under the "Tuned-endpoint synthesis" section
below — additionally call a Vertex AI tuned endpoint (the
auxiliary-synthesis model or a v-series checkpoint) to generate data
that feeds subsequent v-series training.

For the actual tuned models themselves, see:

- `../SFT/main-sequence/v11/` — the eleven-version prediction-model
  sequence (v11 is the final model cited in the paper).
- `../SFT/auxiliary-synthesis/` — the auxiliary data-synthesis model
  described in the paper's §4.6.

## Personal-corpus extractors

- `raw/` — extractors that pull the author's own iMessage, iOS Notes,
  and Gmail archives locally (the paper's §4.2 nomenclature; the on-disk
  extractor uses the `macnotesapp` package to read the iCloud-synced
  Notes store). No raw personal data is checked in; `.gitignore`
  excludes the corpora themselves.

## Utility scripts (base Gemini as a text processor)

- `gemini_clean_dataset.py` — cleaning / normalization.
- `jp_segment_with_gemini.py` — Japanese boundary segmentation.

## Base-Gemini synthesis (superseded)

Early attempts at generating training data with base Gemini alone.
Per the paper §4.6, output from a generic LLM was "correct but
impersonal" — it did not carry the author's voice — and this is
exactly what led the author to build a dedicated auxiliary
data-synthesis model at `../SFT/auxiliary-synthesis/`, which
supersedes both scripts below. **Neither script contributed training
data to any v-series iteration reported in the paper.** They are
retained here for historical reference:

- `generate_slack_replies.py` — generation of synthetic reply pairs
  in Slack conversation style (no real Slack account or thread is
  accessed; the paper excludes business/NDA data channels from any
  pipeline — see §4.2).
- `generate_synthetic_slack.py` — generation of fully synthetic Slack
  conversations (same caveat: no real Slack access).

## Tuned-endpoint synthesis (calls a Vertex AI SFT endpoint)

These call a Vertex AI tuned endpoint — the auxiliary-synthesis model
or a v-series checkpoint — rather than base Gemini. Their outputs
contribute to the training sets of subsequent v-series iterations.

- `synth_v11_tone8.py` — tone-conditioned synthesis (v11 training).
- `synth_v15_notone.py` — ablation without tone conditioning.
- `synth_voice_v2_from_v1.py` — v1 → v2 upgrade pipeline.

## Environment variables

Every script that talks to Vertex AI reads its GCP configuration from
the environment (`PROJECT_ID`, `LOCATION`, `BUCKET_NAME`, and — for
tuned-endpoint scripts — `TUNED_MODEL_ENDPOINT`). Defaults in-source
are placeholders; provide your own values before running.
