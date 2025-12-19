#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import time
import random
import threading
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import difflib

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig


# ========= デフォルト設定 =========

PROJECT_ID_DEFAULT = "project-voice-476504"
LOCATION_DEFAULT = "us-central1"

# v11 tuned model endpoint（指定どおり）
TUNED_MODEL_ENDPOINT_DEFAULT = (
    "projects/700129023625/locations/us-central1/endpoints/2546212743719944192"
)

# seed JSONL (Gemini/Bisonどちらでも)
SEED_JSONL_DEFAULT = (
    "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v11/"
    "voice_boundaryrule_ctxinout_prefix1to3_vark_onebest_gemini__strict_with_userprefix.jsonl"
)

# 出力
OUT_BISON_JSONL_DEFAULT = (
    "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v11/"
    "synth_v11_tone8_bison__strongtone.jsonl"
)
OUT_GEMINI_JSONL_DEFAULT = (
    "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v11/"
    "synth_v11_tone8_gemini__strongtone.jsonl"
)

# seed単位の評価ログ（モデル評価に便利）
SEED_METRICS_JSONL_DEFAULT = (
    "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v11/"
    "synth_v11_tone8_seed_metrics__strongtone.jsonl"
)

MAX_WORKERS_DEFAULT = 4
MAX_MODEL_RETRIES_DEFAULT = -1  # -1: rate系のみ無限バックオフ

# ループ/長文が混ざるのを減らすため、デフォルトは 96 にしておく（必要ならCLIで上書き）
BASE_MAX_OUTPUT_TOKENS_DEFAULT = 96
BASE_TEMPERATURE_DEFAULT = 0.2
DIVERSITY_TEMPERATURE_DEFAULT = 0.85

# v11固定（ヘッダ＋マーカーの間に TONE を差し込む）
FIXED_PROMPT_HEADER = (
    "キーボードの予測変換として[---]に続く言葉を予測変換してください。[---]より前はこれまでのユーザー入力です。\n"
    "ユーザー入力と予測変換の間には境界 [---]を入れてください。"
)
MARKER_LINE = "ーーーー以下が予測変換対象ーーーー"

# ====== トーン 8つ（強化版） ======
# 重要:
# - [---]より前は絶対に変更しない
# - [---]より後のみ / 区切りで出す
# - 繰り返し禁止 / 長すぎ抑制 / トーン差を具体語彙で誘導
CONTEXTS: List[Tuple[str, str]] = [
    (
        "dev",
        "【トーン: dev】開発の文脈で予測変換してください。"
        "開発寄り語彙（例: PR/レビュー/デプロイ/issue/バグ/再現/修正/ログ/確認）を優先。"
        "語尾はフラットで自然（です/ますでも可）。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    (
        "meeting",
        "【トーン: meeting】ミーティングの文脈で予測変換してください。"
        "会議語彙（例: 議題/アジェンダ/共有/確認事項/宿題/決定/進捗/次回）を優先。"
        "結びは『〜します』『〜しましょう』『〜いかがでしょうか』など会議っぽく。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    (
        "casual",
        "【トーン: casual】カジュアルな文脈で予測変換してください。"
        "砕けた口語（例: だよ/だね/しよ/しよう/かな/だと思う）を優先し、敬語はなるべく避ける。"
        "ただし乱暴な表現は避けて自然に。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    (
        "business",
        "【トーン: business】ビジネスの文脈で予測変換してください。"
        "実務的で丁寧（例: 恐れ入りますが/ご確認のほど/差し支えなければ/よろしくお願いいたします）を優先。"
        "長くなりすぎないように [---]より後は原則 20 トークン（/区切りで20個）以内を目安。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    (
        "polite",
        "【トーン: polite】丁寧・敬語の文脈で予測変換してください。"
        "です/ます調＋クッション言葉（例: お手数ですが/恐れ入りますが/ありがとうございます）を優先。"
        "ビジネスほど堅くしすぎず、丁寧さを保った自然文に。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    (
        "friendly",
        "【トーン: friendly】親しみやすい文脈で予測変換してください。"
        "柔らかい語尾（例: 〜ですね/〜だと嬉しいです/〜しよう）や感謝を入れてもよい。"
        "ただし過剰に長くしない。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    (
        "concise",
        "【トーン: concise】短く要点だけの文脈で予測変換してください。"
        "冗長な前置きは避け、[---]より後は原則 8〜12 トークン（/区切りで8〜12個）程度を目安に短く。"
        "敬語は必要最低限に。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    (
        "enthusiastic",
        "【トーン: enthusiastic】明るく前向きな文脈で予測変換してください。"
        "前向き語彙（例: いいですね/助かります/楽しみ/最高/嬉しい）を適度に使い、"
        "必要なら『！』を1個だけ入れてもよい（多用しない）。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
]

# 多様性を促す追記（リトライ時のみ）
DIVERSITY_SUFFIX = (
    "【多様性強化】他のトーン出力と被らないように、[---]より後の「語彙・語尾・敬語レベル・情報量・長さ」を明確に変えてください。"
    "同じ意味や同じ句の繰り返しは禁止です。"
    "長くなりすぎる場合は短くまとめてください。"
    "形式（[---] と / 区切り）と、[---]より前の文字列は必ず維持してください。"
)

# 形式維持リトライ回数（各トーン）
MAX_FORMAT_RETRIES_PER_CONTEXT = 2

# 多様性リトライ回数（各トーン）
MAX_DIVERSITY_RETRIES_PER_CONTEXT = 2

# 多様性判定
MIN_UNIQUE_PER_SEED = 3         # 8本中ユニークがこれ未満なら「多様性弱い」
SIMILARITY_TOO_HIGH = 0.97      # これ以上の類似は「似すぎ」

PROGRESS_BAR_WIDTH = 50

# ========= ここまで =========


# --- 基本ユーティリティ ------------------------------------------------------

def is_rate_error(e: Exception) -> bool:
    msg = str(e).lower()
    keywords = [
        "429", "resource_exhausted", "rate", "quota", "too many requests",
        "temporarily unavailable", "deadline exceeded", "503",
    ]
    return any(k in msg for k in keywords)

def is_not_found_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("404" in msg) or ("not found" in msg)

def print_progress(done: int, total: int) -> None:
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

def _strip_prefix_once(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):].lstrip("\n").lstrip()
    return text

def extract_seed_text(raw_user_text: str) -> str:
    """
    seed側に固定文/文脈/マーカーが混入しても seed 部分だけ抽出して重複を防ぐ。
    - MARKER_LINE があれば「最後の MARKER_LINE 以降」だけ使う
    - なければ固定ヘッダ/文脈が先頭にあれば剥がす
    """
    if not isinstance(raw_user_text, str):
        return ""
    t = raw_user_text.strip()

    if MARKER_LINE in t:
        return t.rsplit(MARKER_LINE, 1)[1].strip()

    # 固定ヘッダを剥がす（完全一致時のみ）
    t = _strip_prefix_once(t, FIXED_PROMPT_HEADER.strip())

    # 文脈プロンプト（完全一致時のみ）を剥がす
    context_prompts = [cp for _, cp in CONTEXTS]
    changed = True
    while changed:
        changed = False
        for cp in context_prompts:
            before = t
            t = _strip_prefix_once(t, cp)
            if t != before:
                changed = True

    # マーカーっぽい行だけ残っているケースを軽く救済
    t = _strip_prefix_once(t, MARKER_LINE)

    return t.strip()

def build_full_input(seed_text: str, tone_prompt: str) -> str:
    """
    TONE は MARKER_LINE の直前に入れる:
      FIXED_PROMPT_HEADER
      tone_prompt
      MARKER_LINE
      seed_text
    """
    seed_text = extract_seed_text(seed_text)
    return f"{FIXED_PROMPT_HEADER}\n{tone_prompt}\n{MARKER_LINE}\n{seed_text}"

def normalize_for_compare(s: str) -> str:
    return " ".join((s or "").strip().split())

def seq_similarity(a: str, b: str) -> float:
    a = normalize_for_compare(a)
    b = normalize_for_compare(b)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def validate_output(seed_text: str, output_text: str) -> Tuple[bool, str]:
    """
    形式維持チェック（厳しめ）:
      - output に [---] が1回
      - seed_text に [---] が1回
      - [---] 前の文脈が一致（完全一致）
      - [---] 後に / が含まれ、空トークンなし（'//'なし）
      - トークンが空白/前後空白を含まない
    """
    if not output_text:
        return False, "empty_output"
    if output_text.count("[---]") != 1:
        return False, "bad_boundary_count_out"
    if seed_text.count("[---]") != 1:
        return False, "bad_boundary_count_seed"

    in_ctx, _ = seed_text.split("[---]", 1)
    out_ctx, out_after = output_text.split("[---]", 1)

    if out_ctx != in_ctx:
        return False, "context_mismatch"
    if "/" not in out_after:
        return False, "no_slash_after"
    if "//" in out_after:
        return False, "double_slash"

    toks = [t for t in out_after.split("/") if t != ""]
    for t in toks:
        if t.strip() == "":
            return False, "whitespace_token"
        if t != t.strip():
            return False, "token_has_edge_space"

    return True, "ok"


# --- レイテンシ統計 ----------------------------------------------------------

def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 1:
        return sorted_vals[-1]
    idx = int((len(sorted_vals) - 1) * p)
    return sorted_vals[idx]


# --- モデル呼び出し（時間ログ込み） ------------------------------------------

def call_model(
    client: genai.Client,
    model_endpoint: str,
    full_input: str,
    config: GenerateContentConfig,
    retries: int,
    base_sleep: float = 0.5,
) -> Tuple[str, Dict[str, Any]]:
    """
    戻り: (text, meta)
    meta:
      - model_attempts
      - latency_sec_total
      - latency_sec_last
    """
    attempt = 0
    last_err: Optional[Exception] = None
    latency_total = 0.0
    latency_last = 0.0

    while True:
        attempt += 1
        t0 = time.time()
        try:
            resp = client.models.generate_content(
                model=model_endpoint,
                contents=full_input,
                config=config,
            )
            latency_last = time.time() - t0
            latency_total += latency_last

            text = (resp.text or "").strip()
            if not text:
                raise RuntimeError("Model returned empty text.")

            meta = {
                "model_attempts": attempt,
                "latency_sec_total": latency_total,
                "latency_sec_last": latency_last,
            }
            return text, meta

        except Exception as e:
            latency_last = time.time() - t0
            latency_total += latency_last

            last_err = e
            rate = is_rate_error(e)
            not_found = is_not_found_error(e)

            print(
                f"[WARN] model call failed (attempt={attempt}, is_rate={rate}, not_found={not_found}, "
                f"latency_last={latency_last:.3f}s): {e}",
                file=sys.stderr,
            )

            if not_found:
                break

            if rate:
                backoff = base_sleep * (2 ** min(attempt, 6))
                backoff += random.uniform(0, 0.5)
                time.sleep(backoff)
                continue

            if retries == -1:
                break
            if attempt >= retries:
                break

            time.sleep(base_sleep * attempt)

    if last_err is None:
        last_err = RuntimeError("Unknown error in call_model.")
    raise last_err


# --- seed 読み込み -----------------------------------------------------------

def load_seeds(seed_path: Path, dedup: bool = True) -> List[str]:
    seeds: List[str] = []
    with seed_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            raw: Optional[str] = None

            # Bison
            if isinstance(obj, dict) and isinstance(obj.get("input_text"), str):
                raw = obj["input_text"]

            # Gemini
            elif isinstance(obj, dict) and "contents" in obj:
                try:
                    contents = obj.get("contents", [])
                    if contents:
                        parts = contents[0].get("parts", [])
                        if parts and isinstance(parts[0].get("text"), str):
                            raw = parts[0]["text"]
                except Exception:
                    raw = None

            if not raw:
                continue

            seed = extract_seed_text(raw)
            if seed:
                seeds.append(seed)

    if not dedup:
        return seeds

    seen = set()
    out = []
    for s in seeds:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# --- 形式維持＋多様性の評価＆必要ならリトライ --------------------------------

def generate_one_tone(
    client: genai.Client,
    model_endpoint: str,
    seed_text: str,
    tone_prompt: str,
    gen_config_base: GenerateContentConfig,
    retries: int,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    1トーン生成：形式維持チェックに通るまでリトライ
    """
    meta: Dict[str, Any] = {
        "format_attempts": 0,
        "format_ok": False,
        "format_fail_reason": None,
        "used_diversity_mode": False,
        "model_attempts": 0,
        "latency_sec_total": 0.0,
        "latency_sec_last": 0.0,
        "diversity_attempts": 0,
        "diversity_best_similarity": 1.0,
    }

    full_input = build_full_input(seed_text, tone_prompt)
    seed_line = extract_seed_text(seed_text)

    for _ in range(MAX_FORMAT_RETRIES_PER_CONTEXT + 1):
        meta["format_attempts"] += 1
        try:
            out, m = call_model(client, model_endpoint, full_input, gen_config_base, retries=retries)
            out = out.strip()
            meta["model_attempts"] += int(m.get("model_attempts", 1))
            meta["latency_sec_total"] += float(m.get("latency_sec_total", 0.0))
            meta["latency_sec_last"] = float(m.get("latency_sec_last", 0.0))
        except Exception as e:
            meta["format_fail_reason"] = f"call_error:{e}"
            continue

        ok, reason = validate_output(seed_line, out)
        if ok:
            meta["format_ok"] = True
            meta["format_fail_reason"] = None
            return out, meta

        meta["format_fail_reason"] = reason

    return None, meta


def improve_diversity_for_tone(
    client: genai.Client,
    model_endpoint: str,
    seed_text: str,
    base_tone_prompt: str,
    existing_outputs: List[str],
    gen_config_diversity: GenerateContentConfig,
    retries: int,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    既存出力に似すぎる場合、温度↑ + 指示強化で多様性リトライ。
    """
    meta: Dict[str, Any] = {
        "diversity_attempts": 0,
        "diversity_ok": False,
        "best_similarity": 1.0,
        "model_attempts": 0,
        "latency_sec_total": 0.0,
    }

    tone_prompt = f"{base_tone_prompt}\n{DIVERSITY_SUFFIX}"
    seed_line = extract_seed_text(seed_text)
    full_input = build_full_input(seed_line, tone_prompt)

    best_out: Optional[str] = None
    best_sim = 1.0

    for _ in range(MAX_DIVERSITY_RETRIES_PER_CONTEXT):
        meta["diversity_attempts"] += 1
        try:
            out, m = call_model(client, model_endpoint, full_input, gen_config_diversity, retries=retries)
            out = out.strip()
            meta["model_attempts"] += int(m.get("model_attempts", 1))
            meta["latency_sec_total"] += float(m.get("latency_sec_total", 0.0))
        except Exception:
            continue

        ok, _ = validate_output(seed_line, out)
        if not ok:
            continue

        sims = [seq_similarity(out, e) for e in existing_outputs] if existing_outputs else [0.0]
        mx = max(sims) if sims else 0.0

        if mx < best_sim:
            best_sim = mx
            best_out = out

        if mx < SIMILARITY_TOO_HIGH:
            meta["diversity_ok"] = True
            meta["best_similarity"] = mx
            return out, meta

    meta["best_similarity"] = best_sim
    return best_out, meta


def process_seed(
    client: genai.Client,
    model_endpoint: str,
    seed_id: int,
    seed_text: str,
    gen_config_base: GenerateContentConfig,
    gen_config_diversity: GenerateContentConfig,
    retries: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[float]]:
    """
    1 seed -> 8トーン生成
    - 形式維持（ダメならリトライ）
    - 多様性不足（ユニーク数/類似度）なら温度↑で再生成
    """
    outputs: Dict[str, Optional[str]] = {}
    metas: Dict[str, Dict[str, Any]] = {}
    latencies: List[float] = []

    # まず形式優先で8本
    for tone_id, tone_prompt in CONTEXTS:
        out, meta = generate_one_tone(
            client, model_endpoint, seed_text, tone_prompt, gen_config_base, retries=retries
        )
        outputs[tone_id] = out
        metas[tone_id] = meta
        latencies.append(float(meta.get("latency_sec_last", 0.0)))

    ok_ids = [tid for tid, o in outputs.items() if o is not None]
    ok_outputs = [outputs[tid] for tid in ok_ids if outputs[tid] is not None]
    unique_n = len({normalize_for_compare(o) for o in ok_outputs if o})

    # 参照（business優先、なければ最初のOK）
    ref_id = "business" if outputs.get("business") else (ok_ids[0] if ok_ids else None)
    ref_out = outputs.get(ref_id) if ref_id else None

    # 多様性不足なら、refに似すぎるものをdiversityで置換
    if unique_n < MIN_UNIQUE_PER_SEED and ref_out is not None:
        for tone_id, tone_prompt in CONTEXTS:
            if tone_id == ref_id:
                continue
            cur = outputs.get(tone_id)
            if cur is None:
                continue
            sim = seq_similarity(cur, ref_out)
            if sim >= SIMILARITY_TOO_HIGH:
                existing = [o for o in outputs.values() if o is not None]
                new_out, dmeta = improve_diversity_for_tone(
                    client,
                    model_endpoint,
                    seed_text,
                    tone_prompt,
                    existing_outputs=existing,
                    gen_config_diversity=gen_config_diversity,
                    retries=retries,
                )
                if new_out is not None:
                    outputs[tone_id] = new_out
                    metas[tone_id]["used_diversity_mode"] = True
                    metas[tone_id]["diversity_attempts"] = int(dmeta.get("diversity_attempts", 0))
                    metas[tone_id]["diversity_best_similarity"] = float(dmeta.get("best_similarity", 1.0))
                    metas[tone_id]["latency_sec_total"] = float(metas[tone_id].get("latency_sec_total", 0.0)) + float(dmeta.get("latency_sec_total", 0.0))
                    latencies.append(float(dmeta.get("latency_sec_total", 0.0)))

        ok_ids = [tid for tid, o in outputs.items() if o is not None]
        ok_outputs = [outputs[tid] for tid in ok_ids if outputs[tid] is not None]
        unique_n = len({normalize_for_compare(o) for o in ok_outputs if o})

    # 類似度統計
    pairs = [(a, b) for i, a in enumerate(ok_outputs) for b in ok_outputs[i + 1:]]
    if pairs:
        sims = [seq_similarity(a, b) for a, b in pairs]
        max_sim = max(sims)
        avg_sim = sum(sims) / len(sims)
    else:
        max_sim = 1.0
        avg_sim = 1.0

    # seedメトリクス
    seed_metrics = {
        "seed_id": seed_id,
        "seed": extract_seed_text(seed_text),
        "ok_tones": len(ok_outputs),
        "unique_outputs": unique_n,
        "max_pairwise_similarity": round(max_sim, 6),
        "avg_pairwise_similarity": round(avg_sim, 6),
        "diversity_weak": bool(unique_n < MIN_UNIQUE_PER_SEED),
        "time_seed_latency_sum_sec": round(sum(float(m.get("latency_sec_total", 0.0)) for m in metas.values()), 6),
    }

    # 行ごとの出力（Bison）
    out_recs: List[Dict[str, Any]] = []
    for tone_id, tone_prompt in CONTEXTS:
        out = outputs.get(tone_id)
        meta = metas.get(tone_id, {})
        if out is None:
            continue

        full_input = build_full_input(seed_text, tone_prompt)

        format_score = 1.0 if meta.get("format_ok") else 0.0
        ok_count = max(1, len(ok_outputs))
        diversity_score = float(unique_n) / float(ok_count)
        total_score = round(0.7 * format_score + 0.3 * (1.0 - max_sim), 6)

        out_recs.append({
            "seed_id": seed_id,
            "seed": extract_seed_text(seed_text),
            "tone_id": tone_id,
            "tone_prompt": tone_prompt,
            "input_text": full_input,
            "output_text": out,
            "status": "ok",

            "latency_sec_last": round(float(meta.get("latency_sec_last", 0.0)), 6),
            "latency_sec_total": round(float(meta.get("latency_sec_total", 0.0)), 6),
            "model_attempts": int(meta.get("model_attempts", 0)),

            "format_ok": bool(meta.get("format_ok", False)),
            "format_fail_reason": meta.get("format_fail_reason"),

            "used_diversity_mode": bool(meta.get("used_diversity_mode", False)),
            "diversity_attempts": int(meta.get("diversity_attempts", 0)),
            "diversity_best_similarity": float(meta.get("diversity_best_similarity", 1.0)),

            "unique_outputs_in_seed": unique_n,
            "max_pairwise_similarity_in_seed": round(max_sim, 6),
            "avg_pairwise_similarity_in_seed": round(avg_sim, 6),

            "format_score": format_score,
            "diversity_score": round(diversity_score, 6),
            "total_score": total_score,
        })

    return out_recs, seed_metrics, latencies


# --- メイン -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="v11 tuned modelで合成（トーン8 / 形式維持＋多様性評価＋生成時間ログ＋必要ならリトライ）"
    )
    ap.add_argument("--project-id", default=PROJECT_ID_DEFAULT)
    ap.add_argument("--location", default=LOCATION_DEFAULT)
    ap.add_argument("--model-endpoint", default=TUNED_MODEL_ENDPOINT_DEFAULT)
    ap.add_argument("--seed-jsonl", default=SEED_JSONL_DEFAULT)
    ap.add_argument("--out-bison", default=OUT_BISON_JSONL_DEFAULT)
    ap.add_argument("--out-gemini", default=OUT_GEMINI_JSONL_DEFAULT)
    ap.add_argument("--seed-metrics-jsonl", default=SEED_METRICS_JSONL_DEFAULT)
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS_DEFAULT)
    ap.add_argument("--retries", type=int, default=MAX_MODEL_RETRIES_DEFAULT)
    ap.add_argument("--limit-seeds", type=int, default=0)
    ap.add_argument("--no-seed-dedup", action="store_true")
    ap.add_argument("--report-every", type=int, default=50)
    ap.add_argument("--random-seed", type=int, default=20251219)

    # 生成パラメータをCLIで調整できるように
    ap.add_argument("--max-output-tokens", type=int, default=BASE_MAX_OUTPUT_TOKENS_DEFAULT)
    ap.add_argument("--temperature", type=float, default=BASE_TEMPERATURE_DEFAULT)
    ap.add_argument("--diversity-temperature", type=float, default=DIVERSITY_TEMPERATURE_DEFAULT)

    args = ap.parse_args()

    random.seed(args.random_seed)

    # 生成設定（基本 / 多様性）
    gen_config_base = GenerateContentConfig(
        temperature=float(args.temperature),
        max_output_tokens=int(args.max_output_tokens),
        thinking_config=ThinkingConfig(thinking_budget=0),
    )
    gen_config_diversity = GenerateContentConfig(
        temperature=float(args.diversity_temperature),
        max_output_tokens=int(args.max_output_tokens),
        thinking_config=ThinkingConfig(thinking_budget=0),
    )

    seed_path = Path(args.seed_jsonl)
    out_bison_path = Path(args.out_bison)
    out_gemini_path = Path(args.out_gemini)
    seed_metrics_path = Path(args.seed_metrics_jsonl)

    if not seed_path.exists():
        print(f"[ERROR] seed JSONL not found: {seed_path}", file=sys.stderr)
        sys.exit(1)

    out_bison_path.parent.mkdir(parents=True, exist_ok=True)
    out_gemini_path.parent.mkdir(parents=True, exist_ok=True)
    seed_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = load_seeds(seed_path, dedup=(not args.no_seed_dedup))
    if args.limit_seeds and args.limit_seeds > 0:
        seeds = seeds[: args.limit_seeds]
    if not seeds:
        print(f"[ERROR] no seeds found in {seed_path}", file=sys.stderr)
        sys.exit(1)

    total = len(seeds)
    print(f"[INFO] seeds to process={total} (tones={len(CONTEXTS)})", file=sys.stderr)
    print(f"[INFO] model endpoint default={TUNED_MODEL_ENDPOINT_DEFAULT}", file=sys.stderr)
    print(
        f"[INFO] gen_config: temp={args.temperature}, div_temp={args.diversity_temperature}, "
        f"max_output_tokens={args.max_output_tokens}",
        file=sys.stderr,
    )

    client = genai.Client(vertexai=True, project=args.project_id, location=args.location)

    f_bison = out_bison_path.open("a", encoding="utf-8")
    f_gemini = out_gemini_path.open("a", encoding="utf-8")
    f_seedm = seed_metrics_path.open("a", encoding="utf-8")
    lock = threading.Lock()

    done = 0
    print_progress(done, total)

    # 集計
    all_latencies: List[float] = []
    seeds_written = 0
    lines_written = 0
    seeds_div_weak = 0
    format_ok_lines = 0
    total_lines = 0
    total_score_sum = 0.0

    max_workers = max(1, min(args.max_workers, 50))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for seed_id, seed_text in enumerate(seeds):
            futures.append(
                ex.submit(
                    process_seed,
                    client,
                    args.model_endpoint,
                    seed_id,
                    seed_text,
                    gen_config_base,
                    gen_config_diversity,
                    args.retries,
                )
            )

        for fut in as_completed(futures):
            recs, seedm, lat_list = fut.result()
            all_latencies.extend([x for x in lat_list if x and x > 0])

            with lock:
                f_seedm.write(json.dumps(seedm, ensure_ascii=False) + "\n")
                f_seedm.flush()

            if seedm.get("diversity_weak"):
                seeds_div_weak += 1

            if not recs:
                done += 1
                print_progress(done, total)
                continue

            with lock:
                for r in recs:
                    f_bison.write(json.dumps(r, ensure_ascii=False) + "\n")
                    gemini_line = {
                        "contents": [
                            {"role": "user", "parts": [{"text": r["input_text"]}]},
                            {"role": "model", "parts": [{"text": r["output_text"]}]},
                        ]
                    }
                    f_gemini.write(json.dumps(gemini_line, ensure_ascii=False) + "\n")
                f_bison.flush()
                f_gemini.flush()

            seeds_written += 1
            lines_written += len(recs)

            for r in recs:
                total_lines += 1
                if r.get("format_ok"):
                    format_ok_lines += 1
                total_score_sum += float(r.get("total_score", 0.0))

            done += 1

            if args.report_every > 0 and done % args.report_every == 0:
                lat_sorted = sorted(all_latencies)

                def _p(q: float) -> float:
                    if not lat_sorted:
                        return 0.0
                    return lat_sorted[int((len(lat_sorted) - 1) * q)]

                p50 = _p(0.50)
                p90 = _p(0.90)
                p99 = _p(0.99)
                fmt_rate = (format_ok_lines / total_lines) if total_lines else 0.0
                avg_score = (total_score_sum / total_lines) if total_lines else 0.0
                print(
                    f"\n[LOG] done={done}/{total} "
                    f"seeds_written={seeds_written} lines={lines_written} "
                    f"format_ok_rate={fmt_rate:.4f} "
                    f"div_weak_seeds={seeds_div_weak} "
                    f"latency_p50={p50:.3f}s p90={p90:.3f}s p99={p99:.3f}s "
                    f"avg_total_score={avg_score:.4f}",
                    file=sys.stderr,
                )

            print_progress(done, total)

    f_bison.close()
    f_gemini.close()
    f_seedm.close()

    lat_sorted = sorted(all_latencies)
    p50 = percentile(lat_sorted, 0.50)
    p90 = percentile(lat_sorted, 0.90)
    p99 = percentile(lat_sorted, 0.99)
    fmt_rate = (format_ok_lines / total_lines) if total_lines else 0.0
    avg_score = (total_score_sum / total_lines) if total_lines else 0.0

    print(
        "\n[REPORT]\n"
        f"  seeds_processed: {total}\n"
        f"  seeds_written  : {seeds_written}\n"
        f"  lines_written  : {lines_written}\n"
        f"  tones          : {len(CONTEXTS)}\n"
        f"  diversity_weak_seeds: {seeds_div_weak}\n"
        f"  format_ok_rate (line): {fmt_rate:.6f}\n"
        f"  avg_total_score (line): {avg_score:.6f}\n"
        f"  latency_sec p50/p90/p99: {p50:.3f}s / {p90:.3f}s / {p99:.3f}s\n"
        f"\n[INFO] outputs:\n"
        f"  Bison       : {out_bison_path}\n"
        f"  Gemini      : {out_gemini_path}\n"
        f"  Seed metrics: {seed_metrics_path}\n",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
