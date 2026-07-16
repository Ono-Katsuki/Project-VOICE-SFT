# Project-VOICE-SFT

**Project VOICE (Valuing Our Individual Communication Expression)** is an
open-source project published by Google (documented in the paper's
Project VOICE footnote). This repository is an independent personal
repository maintained by the author — it is not Google-affiliated code.
It contains the SFT pipeline the author uses within Project VOICE to
specialize a Gemini 2.5 Flash predictive-text model for his own
single-switch input environment. The resulting Vertex AI endpoint powers
the side-by-side comparison demo at
[Ono-Katsuki/tone-comparison](https://github.com/Ono-Katsuki/tone-comparison).
The parallel on-device Gemma 3 270M SFT+DPO experiment discussed in
the paper lives in a separate notebook shipped with the paper's
supplementary bundle.

This repository is a companion artifact to the following paper:

> Katsuki Ono. 2026. **Reconfiguration of Ability: An Autoethnography of
> LLM-Mediated Development by a Developer with Severe Physical Disability.**
> In *The 28th International ACM SIGACCESS Conference on Computers and
> Accessibility (ASSETS '26).* ACM. https://doi.org/10.1145/3797867.3828990

See `CITATION.cff` for a machine-readable citation.

> **Note on model lifecycle.** The `gemini-2.5-flash` identifier
> targeted throughout this repository is scheduled for shutdown on
> **October 16, 2026** (paper §5.8; Gemini API deprecations page).
> Treat the code here as a reference for the pipeline logic and
> structure — the same flow ports directly to whichever Gemini release
> is currently supported at the time you run it.

## Layout

```
Project-VOICE-SFT/
├── SFT/
│   ├── tune_gemini25_flash_e2e.py   # Vertex AI Gemini 2.5 Flash SFT launcher (canonical)
│   ├── README.md                    # Launcher usage notes
│   ├── main-sequence/               # v1/ … v11/ — the eleven-version prediction-model
│   │                                #   sequence reported in the paper. v6–v11 are the
│   │                                #   paper's "Long Refinement" phase (§4.7); see the
│   │                                #   paper for the choices that characterize it.
│   │                                #   v11 is the final model, used in the demo and
│   │                                #   cited in the paper.
│   ├── auxiliary-synthesis/         # An auxiliary data-synthesis model, separate from
│   │                                #   the v1–v11 sequence, used to generate part of
│   │                                #   the training data for v11.
│   └── post-paper/                  # v12/ … v17/ — later iterations produced after
│                                    #   the paper.
└── dataset/
    ├── raw/                         # Personal source extractors for local execution
    ├── generate_slack_replies.py    # Synthetic reply-pair generation, Slack-style
    │                                #   (superseded; no real Slack access — see
    │                                #   dataset/README.md)
    ├── generate_synthetic_slack.py  # Synthetic Slack-style conversation generation
    │                                #   (superseded; no real Slack access)
    ├── gemini_clean_dataset.py      # Gemini-assisted cleaning
    ├── jp_segment_with_gemini.py    # Japanese boundary segmentation
    ├── synth_v11_tone8.py           # Tone-conditioned synthesis (used for v11)
    ├── synth_v15_notone.py          # Ablation without tone conditioning
    └── synth_voice_v2_from_v1.py    # v1 → v2 upgrade pipeline
```

The version referenced in the paper is **v11** — a Gemini 2.5 Flash SFT run
conditioned on eight tone labels (`dev / meeting / casual / business / polite /
friendly / concise / enthusiastic`). The deployed keyboard shows one
candidate per tone at inference (paper §4.8); the companion demo repository
[`tone-comparison`](https://github.com/Ono-Katsuki/tone-comparison)
additionally diversity-selects three suggestions for side-by-side viewing.

## Data provenance and privacy

The scripts under `dataset/raw/` extract training material from the
**author's own personal communication corpora** — executed locally against
the author's own accounts. **No raw personal data is checked into this
repository** — `.gitignore` excludes `dataset/messages.jsonl`,
`dataset/.seen_ids.txt`, and `dataset/attachments/`. Only the extraction,
generation, and cleaning code is public.

If you want to reproduce a run against your own data, point the extractors
at your own accounts and re-run the synthesis pipeline. The tuned Vertex AI
endpoint produced by this repo — and separately, the on-device Gemma 3
270M model on Hugging Face that comes out of the supplementary-bundle
notebook — are both personalized to a single individual and are not
intended for cross-user deployment.

## Setup

```bash
pip install -r requirements.txt
gcloud auth application-default login
```

Then edit the configuration block at the top of the target
`SFT/tune_gemini25_flash_e2e.py` (or a specific version's `train.py`):

| Variable | Example |
|---|---|
| `PROJECT_ID` | Your GCP project ID |
| `LOCATION` | Tuning-supported region (e.g. `us-central1`) |
| `BUCKET_NAME` | GCS bucket you own |
| `GCS_PREFIX` | Any path prefix |
| `LOCAL_TRAIN_FILE` / `LOCAL_VALID_FILE` | Local JSONL paths |
| `TUNED_MODEL_DISPLAY_NAME` | Human-readable display name |

Then:

```bash
python SFT/tune_gemini25_flash_e2e.py
```

The v11 launcher (`SFT/main-sequence/v11/train.py`) additionally
estimates costs up front and refuses to submit above `MAX_BUDGET_JPY`
(default ¥50,000) — override via the constants at the top of that file.
The canonical `tune_gemini25_flash_e2e.py` submits the job directly
without the cost guard.

## Cited BibTeX

```bibtex
@inproceedings{ono2026reconfiguration,
  author    = {Katsuki Ono},
  title     = {Reconfiguration of Ability: An Autoethnography of {LLM}-Mediated
               Development by a Developer with Severe Physical Disability},
  booktitle = {The 28th International ACM SIGACCESS Conference on Computers
               and Accessibility (ASSETS '26)},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3797867.3828990},
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
