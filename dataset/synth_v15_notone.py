#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import time
import random
import threading
import re
import difflib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from collections import Counter

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig


# ========= デフォルト設定 =========

PROJECT_ID_DEFAULT = "project-voice-476504"
LOCATION_DEFAULT = "us-central1"

# ★v15 tuned endpoint（CLIで上書き可）
TUNED_MODEL_ENDPOINT_DEFAULT = (
    "projects/700129023625/locations/us-central1/endpoints/2363280397137084416"
)

# seedはv11のまま
SEED_JSONL_DEFAULT = (
    "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v11/"
    "voice_boundaryrule_ctxinout_prefix1to3_vark_onebest_gemini__strict_with_userprefix.jsonl"
)

# 出力（v15）
OUT_BISON_JSONL_DEFAULT = (
    "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v15/"
    "synth_v15_notone_bison.jsonl"
)
OUT_GEMINI_JSONL_DEFAULT = (
    "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v15/"
    "synth_v15_notone_gemini.jsonl"
)
SEED_METRICS_JSONL_DEFAULT = (
    "/Users/onokatsuki/Documents/GitHub/Project-VOICE-SFT/SFT/v15/"
    "synth_v15_notone_seed_metrics.jsonl"
)

MAX_WORKERS_DEFAULT = 4
MAX_MODEL_RETRIES_DEFAULT = -1  # -1: rate系のみ無限バックオフ

DEFAULT_MAX_OUTPUT_TOKENS = 260
DEFAULT_TEMPERATURE = 0.35
DEFAULT_FORMAT_RETRIES = 3
DEFAULT_DIVERSITY_RETRIES = 2

# 多様性/ループ抑制（現実的に通る強さ）
DEFAULT_MAX_AFTER_TOKS = 24
DEFAULT_MAX_SENT_SIM = 0.95
DEFAULT_MIN_UNIQUE_SENT = 3  # after部が最低3ユニーク

# --- seed 側に混ざる v11 prompt（フィルタ用） ---
FIXED_PROMPT_HEADER_V11 = (
    "キーボードの予測変換として[---]に続く言葉を予測変換してください。[---]より前はこれまでのユーザー入力です。\n"
    "ユーザー入力と予測変換の間には境界 [---]を入れてください。"
)

# --- 生成に使う v15 prompt（words=8固定） ---
FIXED_PROMPT_HEADER_V15 = (
    "キーボードの予測変換として[---]に続く異なる言葉を4つ予測変換してください。[---]より前はこれまでのユーザー入力です。\n"
    "ユーザー入力と予測変換の間には境界 [---]を入れてください。\n"
    "【重要】[---]より前の文字列は1文字も変更せず、そのまま出力してください。\n\n"
    "またキーボードの予測変換として[---]に続く異なる単語を8つ予測変換してください。\n"
    "sentences の各要素は、[---]より後を / 区切りで出力してください。\n\n"
    "{\"sentences\":[...4], \"words\":[...8]} 形式で答えてください。"
)

# ★多様性を促す指示（“通る”レベル）
DIVERSITY_INSTRUCTION = (
    "【多様性制約】sentencesの4本は、意味・話題・文末表現をできるだけ変えてください。\n"
    "例：断定 / 疑問 / 依頼 / 提案 / カジュアル などを混ぜる。\n"
    "禁止：4本がほぼ同じ言い回し（語尾だけ違う等）。\n"
    "『と/思う』『と思います』は4本中2回まで。\n"
)

# 不適切寄り話題の禁止（IME用途向け）
SENSITIVE_BAN_INSTRUCTION = (
    "【内容制約】自傷・死・暴力・性的な内容、差別、個人情報の要求、過度に不快な内容は禁止。"
    "そのような話題に寄らず、日常・仕事・学習・連絡など安全な内容で作成。\n"
)

# リトライ時の強制（JSON/words）
STRICT_OUTPUT_SUFFIX = (
    "【出力制約】返答はJSONオブジェクト1個のみ。前後に説明文・Markdown・コードブロックを付けない。\n"
    "sentencesは必ず長さ4。\n"
    "wordsは必ず長さ8（ちょうど8）で、重複なし。words内に[---]や/や空白を含めない。\n"
)

MARKER_LINE = "ーーーー以下が予測変換対象ーーーー"
PROGRESS_BAR_WIDTH = 50

# センシティブ検知（IME向けに粗く弾く）
SENSITIVE_PAT = re.compile(
    r"(死|自殺|死ん|殺|殺し|首つ|首吊|リスカ|希死|遺書|暴力|レイプ|強姦|性行為|エロ|AV|ポルノ|裸|児童|虐待)",
    re.IGNORECASE,
)


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
    seedファイルに v11/v15 prompt が混ざっていても、
    最終的に seed 部分（例: '...context...[---]やま'）だけを返す。
    """
    if not isinstance(raw_user_text, str):
        return ""
    t = raw_user_text.strip()

    if MARKER_LINE in t:
        t = t.rsplit(MARKER_LINE, 1)[1].strip()

    changed = True
    while changed:
        before = t
        t = _strip_prefix_once(t, FIXED_PROMPT_HEADER_V11.strip())
        t = _strip_prefix_once(t, FIXED_PROMPT_HEADER_V15.strip())
        t = _strip_prefix_once(t, DIVERSITY_INSTRUCTION.strip())
        t = _strip_prefix_once(t, SENSITIVE_BAN_INSTRUCTION.strip())
        t = _strip_prefix_once(t, STRICT_OUTPUT_SUFFIX.strip())
        t = _strip_prefix_once(t, MARKER_LINE)
        changed = (t != before)

    return t.strip()

def build_full_input(seed_text: str, extra_instruction: str = "", ban_sensitive: bool = True) -> str:
    seed_text = extract_seed_text(seed_text)
    parts = [FIXED_PROMPT_HEADER_V15, DIVERSITY_INSTRUCTION]
    if ban_sensitive:
        parts.append(SENSITIVE_BAN_INSTRUCTION)
    if extra_instruction.strip():
        parts.append(extra_instruction.strip())
    parts.append(MARKER_LINE)
    parts.append(seed_text)
    return "\n".join(parts)

def parse_json_lenient(text: str) -> Dict[str, Any]:
    """
    - ```json ... ``` のフェンスを剥がす
    - 先頭/末尾の余計な文字が混ざっても最初の {...} を抽出して読む
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("empty")

    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t).strip()
        if t.endswith("```"):
            t = t[:-3].strip()

    try:
        return json.loads(t)
    except Exception:
        pass

    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        return json.loads(t[s:e+1])

    raise ValueError("json_parse_failed")

def normalized_json_text(obj: Dict[str, Any]) -> str:
    out = {
        "sentences": obj.get("sentences", []),
        "words": obj.get("words", []),
    }
    return json.dumps(out, ensure_ascii=False)

def _after_tokens(sentence: str) -> List[str]:
    if "[---]" not in sentence:
        return []
    after = sentence.split("[---]", 1)[1]
    return [t for t in after.split("/") if t != ""]

def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def hint_for_reason(reason: Optional[str], max_after_toks: int, max_sent_sim: float) -> str:
    if not reason:
        return ""
    if reason == "json_parse_failed":
        return "前回はJSONとして解釈できませんでした。必ずJSONオブジェクト1個だけを返し、先頭は{、末尾は}で終えてください。"
    if reason == "words_has_duplicates":
        return "前回wordsに重複がありました。wordsは必ず8個すべて異なる単語にしてください。"
    if reason == "bad_words_len_not_8":
        return "前回wordsの個数が8ではありませんでした。wordsは必ず8個ちょうどにしてください。"
    if reason and reason.startswith("sentence_") and "len_gt_" in reason:
        return f"sentences各要素の[---]以降は / 区切りで最大{max_after_toks}トークン以内にしてください。"
    if reason == "sentences_too_similar":
        return f"4文が似すぎています。語彙・語尾・話題を変えてください（語尾だけ変更は禁止）。類似度は{max_sent_sim}未満を目標。"
    if reason == "sentences_not_diverse":
        return "4文の内容が似ています。話題や文末（断定/疑問/依頼/提案/カジュアル）を混ぜてください。"
    if reason == "thought_phrase_too_many":
        return "『思う/思います』の多用は禁止。4文中2回までにしてください。"
    if reason == "sensitive_in_sentences" or reason == "sensitive_in_words":
        return "センシティブな話題は禁止です。安全な日常・仕事・学習・連絡の内容にしてください。"
    return ""


# --- 出力検証（形式＋多様性＋反復抑制） --------------------------------------

def validate_output_json(
    seed_text: str,
    output_text: str,
    require_words8: bool = True,
    max_after_toks: int = DEFAULT_MAX_AFTER_TOKS,
    max_sent_sim: float = DEFAULT_MAX_SENT_SIM,
    ban_sensitive: bool = True,
    min_unique_sent: int = DEFAULT_MIN_UNIQUE_SENT,
) -> Tuple[bool, str, Dict[str, Any], Optional[Dict[str, Any]]]:
    info: Dict[str, Any] = {}
    if not output_text:
        return False, "empty_output", info, None

    try:
        obj = parse_json_lenient(output_text)
    except Exception:
        return False, "json_parse_failed", info, None

    if not isinstance(obj, dict):
        return False, "json_not_object", info, None

    sentences = obj.get("sentences")
    words = obj.get("words")

    if not isinstance(sentences, list) or len(sentences) != 4:
        return False, "bad_sentences_len", info, obj
    if not isinstance(words, list):
        return False, "bad_words_type", info, obj
    if require_words8 and len(words) != 8:
        return False, "bad_words_len_not_8", info, obj
    if (not require_words8) and len(words) > 8:
        return False, "bad_words_len_gt_8", info, obj

    seed = extract_seed_text(seed_text)
    if seed.count("[---]") != 1:
        return False, "bad_seed_boundary_count", info, obj
    seed_ctx, _ = seed.split("[---]", 1)

    # words check
    seen = set()
    clean_words: List[str] = []
    for i, w in enumerate(words):
        if not isinstance(w, str):
            return False, f"word_{i}_not_str", info, obj
        if not w.strip():
            return False, f"word_{i}_empty", info, obj
        if w != w.strip():
            return False, f"word_{i}_edge_space", info, obj
        if "[---]" in w or MARKER_LINE in w:
            return False, f"word_{i}_has_boundary_or_marker", info, obj
        if "/" in w or " " in w:
            return False, f"word_{i}_has_slash_or_space", info, obj
        if w in seen:
            return False, "words_has_duplicates", info, obj
        if ban_sensitive and SENSITIVE_PAT.search(w):
            return False, "sensitive_in_words", info, obj
        seen.add(w)
        clean_words.append(w)

    after_norms: List[str] = []
    after_lens: List[int] = []
    thought_cnt = 0

    for i, s in enumerate(sentences):
        if not isinstance(s, str) or not s.strip():
            return False, f"sentence_{i}_empty", info, obj
        if s.count("[---]") != 1:
            return False, f"sentence_{i}_bad_boundary_count", info, obj

        s_ctx, _ = s.split("[---]", 1)
        if s_ctx != seed_ctx:
            return False, f"sentence_{i}_context_mismatch", info, obj

        if ban_sensitive and SENSITIVE_PAT.search(s):
            return False, "sensitive_in_sentences", info, obj

        toks = _after_tokens(s)
        L = len(toks)
        if L < 1:
            return False, f"sentence_{i}_len_lt1", info, obj
        if L > max_after_toks:
            return False, f"sentence_{i}_len_gt_{max_after_toks}", info, obj

        # 『思う』は多用しがちなので上限を設ける（緩め）
        thought_cnt += sum(1 for t in toks if t in ("思う", "思います"))

        # token反復：4回以上でNG（緩め）
        c = Counter(toks)
        if c and max(c.values()) >= 4:
            return False, f"sentence_{i}_token_repeat", info, obj

        # bigram反復：長めの時だけ見て 3回以上でNG（緩め）
        if L >= 16:
            bigrams = Counter(tuple(toks[j:j+2]) for j in range(L - 1))
            if bigrams and max(bigrams.values()) >= 3:
                return False, f"sentence_{i}_bigram_repeat", info, obj

        after_norms.append(" ".join(toks))
        after_lens.append(L)

    if thought_cnt >= 3:
        return False, "thought_phrase_too_many", info, obj

    # 4文が“ほぼ同じ”を避ける：ユニーク数が最低min_unique_sent
    if len(set(after_norms)) < min_unique_sent:
        return False, "sentences_not_diverse", info, obj

    # 長さバリエーションがゼロ（全部同じ長さ）ならNG（弱い制約）
    if len(set(after_lens)) < 2:
        return False, "sentences_length_not_diverse", info, obj

    # sentences 同士の類似度を抑制（現実的：0.95）
    for i in range(4):
        for j in range(i + 1, 4):
            if _sim(after_norms[i], after_norms[j]) > max_sent_sim:
                return False, "sentences_too_similar", info, obj

    info["sentences_count"] = 4
    info["words_count"] = len(clean_words)
    return True, "ok", info, obj


# --- モデル呼び出し（rateのみバックオフ） ------------------------------------

def make_config(temp: float, max_tokens: int) -> GenerateContentConfig:
    return GenerateContentConfig(
        temperature=float(temp),
        max_output_tokens=int(max_tokens),
        thinking_config=ThinkingConfig(thinking_budget=0),
    )

def call_model_once(
    client: genai.Client,
    model_endpoint: str,
    full_input: str,
    config: GenerateContentConfig,
    retries: int,
    base_sleep: float = 0.5,
) -> Tuple[str, Dict[str, Any]]:
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

    raise (last_err or RuntimeError("Unknown error in call_model_once."))


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


# --- resume 用：既に処理済み seed をスキップ ---------------------------------

def load_processed_seeds_from_bison(path: Path) -> Set[str]:
    processed: Set[str] = set()
    if not path.exists():
        return processed
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            seed = obj.get("seed")
            if isinstance(seed, str) and seed:
                processed.add(seed)
    return processed


# --- seed 1本処理（生成＋評価＋リトライ） -------------------------------------

def generate_with_retries(
    client: genai.Client,
    model_endpoint: str,
    seed_line: str,
    temperature: float,
    max_output_tokens: int,
    retries: int,
    format_retries: int,
    diversity_retries: int,
    require_words8: bool,
    max_after_toks: int,
    max_sent_sim: float,
    min_unique_sent: int,
    ban_sensitive: bool,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    - format_retries: 形式が崩れた時のリトライ
    - diversity_retries: 形式OKだが多様性NGの時の追加リトライ
    - 失敗reasonを次の入力にフィードバックして改善
    """
    meta: Dict[str, Any] = {
        "attempts_total": 0,
        "format_attempts": 0,
        "diversity_attempts": 0,
        "ok": False,
        "fail_reason": None,
        "model_attempts": 0,
        "latency_sec_total": 0.0,
        "latency_sec_last": 0.0,
    }

    last_reason: Optional[str] = None

    max_tries = (format_retries + 1) + diversity_retries

    for k in range(max_tries):
        meta["attempts_total"] += 1
        if k <= format_retries:
            meta["format_attempts"] += 1
        else:
            meta["diversity_attempts"] += 1

        extra = ""
        temp_k = temperature

        if k >= 1:
            extra = STRICT_OUTPUT_SUFFIX + f"（再試行{k}回目：形式と多様性を厳守）\n"
            extra += hint_for_reason(last_reason, max_after_toks, max_sent_sim)

            # ★形式NGは温度↓、多様性NGは温度↑
            if k <= format_retries:
                temp_k = max(0.10, temperature * 0.75)
            else:
                temp_k = min(0.85, temperature * 1.15)

        full_input = build_full_input(seed_line, extra_instruction=extra, ban_sensitive=ban_sensitive)
        cfg = make_config(temp_k, max_output_tokens)

        try:
            out, m = call_model_once(client, model_endpoint, full_input, cfg, retries=retries)
            meta["model_attempts"] += int(m.get("model_attempts", 1))
            meta["latency_sec_total"] += float(m.get("latency_sec_total", 0.0))
            meta["latency_sec_last"] = float(m.get("latency_sec_last", 0.0))
        except Exception as e:
            last_reason = f"call_error:{e}"
            meta["fail_reason"] = last_reason
            continue

        ok, reason, _, parsed = validate_output_json(
            seed_line,
            out,
            require_words8=require_words8,
            max_after_toks=max_after_toks,
            max_sent_sim=max_sent_sim,
            ban_sensitive=ban_sensitive,
            min_unique_sent=min_unique_sent,
        )

        if ok and parsed is not None:
            meta["ok"] = True
            meta["fail_reason"] = None
            return normalized_json_text(parsed), meta

        last_reason = reason
        meta["fail_reason"] = reason

    return None, meta


def process_seed_once(
    client: genai.Client,
    model_endpoint: str,
    seed_id: int,
    seed_text: str,
    temperature: float,
    max_output_tokens: int,
    retries: int,
    format_retries: int,
    diversity_retries: int,
    require_words8: bool,
    max_after_toks: int,
    max_sent_sim: float,
    min_unique_sent: int,
    ban_sensitive: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    seed_line = extract_seed_text(seed_text)

    out_text, meta = generate_with_retries(
        client=client,
        model_endpoint=model_endpoint,
        seed_line=seed_line,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        retries=retries,
        format_retries=format_retries,
        diversity_retries=diversity_retries,
        require_words8=require_words8,
        max_after_toks=max_after_toks,
        max_sent_sim=max_sent_sim,
        min_unique_sent=min_unique_sent,
        ban_sensitive=ban_sensitive,
    )

    status = "ok" if out_text is not None else "fail"
    ok_flag = bool(meta.get("ok", False))
    fail_reason = meta.get("fail_reason")

    sentences_count = 0
    words_count = 0
    if out_text is not None:
        try:
            obj = json.loads(out_text)
            sentences_count = len(obj.get("sentences", []) or [])
            words_count = len(obj.get("words", []) or [])
        except Exception:
            pass

    full_input_for_log = build_full_input(seed_line, ban_sensitive=ban_sensitive)

    rec = {
        "seed_id": seed_id,
        "seed": seed_line,
        "input_text": full_input_for_log,
        "output_text": out_text,
        "status": status,

        "format_ok": ok_flag,
        "format_fail_reason": None if ok_flag else fail_reason,

        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
        "format_retries": int(format_retries),
        "diversity_retries": int(diversity_retries),
        "require_words8": bool(require_words8),

        "max_after_toks": int(max_after_toks),
        "max_sent_sim": float(max_sent_sim),
        "min_unique_sent": int(min_unique_sent),
        "ban_sensitive": bool(ban_sensitive),

        "attempts_total": int(meta.get("attempts_total", 0)),
        "format_attempts": int(meta.get("format_attempts", 0)),
        "diversity_attempts": int(meta.get("diversity_attempts", 0)),
        "model_attempts": int(meta.get("model_attempts", 0)),
        "latency_sec_last": round(float(meta.get("latency_sec_last", 0.0)), 6),
        "latency_sec_total": round(float(meta.get("latency_sec_total", 0.0)), 6),

        "sentences_count": int(sentences_count),
        "words_count": int(words_count),
    }

    seedm = {
        "seed_id": seed_id,
        "seed": seed_line,
        "status": status,
        "format_ok": ok_flag,
        "format_fail_reason": rec["format_fail_reason"],
        "sentences_count": int(sentences_count),
        "words_count": int(words_count),
        "attempts_total": rec["attempts_total"],
        "format_attempts": rec["format_attempts"],
        "diversity_attempts": rec["diversity_attempts"],
        "model_attempts": rec["model_attempts"],
        "latency_sec_last": rec["latency_sec_last"],
        "latency_sec_total": rec["latency_sec_total"],
    }

    return rec, seedm


# --- メイン -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="v15 tuned modelで合成（トーン無し / seedごと生成 / JSON形式 + 多様性 + 反復抑制 + リトライ）"
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
    ap.add_argument("--report-every", type=int, default=200)
    ap.add_argument("--random-seed", type=int, default=20251221)

    ap.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)

    ap.add_argument("--format-retries", type=int, default=DEFAULT_FORMAT_RETRIES)
    ap.add_argument("--diversity-retries", type=int, default=DEFAULT_DIVERSITY_RETRIES)

    ap.add_argument("--max-after-toks", type=int, default=DEFAULT_MAX_AFTER_TOKS)
    ap.add_argument("--max-sent-sim", type=float, default=DEFAULT_MAX_SENT_SIM)
    ap.add_argument("--min-unique-sent", type=int, default=DEFAULT_MIN_UNIQUE_SENT)

    ap.add_argument("--allow-words-le8", action="store_true")
    ap.add_argument("--ban-sensitive-off", action="store_true")

    ap.add_argument("--resume-skip-existing", action="store_true")
    ap.add_argument("--only-ok", action="store_true")

    args = ap.parse_args()
    random.seed(args.random_seed)

    require_words8 = (not args.allow_words_le8)
    ban_sensitive = (not args.ban_sensitive_off)

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

    if args.resume_skip_existing:
        processed = load_processed_seeds_from_bison(out_bison_path)
        print(f"[INFO] resume: existing seeds={len(processed)}", file=sys.stderr)
        seeds = [s for s in seeds if extract_seed_text(s) not in processed]

    total = len(seeds)
    print(f"[INFO] seeds to process={total}", file=sys.stderr)
    print(f"[INFO] model endpoint={args.model_endpoint}", file=sys.stderr)
    print(
        f"[INFO] temperature={args.temperature} max_output_tokens={args.max_output_tokens} "
        f"format_retries={args.format_retries} diversity_retries={args.diversity_retries} "
        f"max_after_toks={args.max_after_toks} max_sent_sim={args.max_sent_sim} "
        f"min_unique_sent={args.min_unique_sent} require_words8={require_words8} "
        f"ban_sensitive={ban_sensitive} max_workers={args.max_workers}",
        file=sys.stderr,
    )

    client = genai.Client(vertexai=True, project=args.project_id, location=args.location)

    f_bison = out_bison_path.open("a", encoding="utf-8")
    f_gemini = out_gemini_path.open("a", encoding="utf-8")
    f_seedm = seed_metrics_path.open("a", encoding="utf-8")
    lock = threading.Lock()

    done = 0
    print_progress(done, total)

    ok_count = 0
    latencies: List[float] = []

    max_workers = max(1, min(args.max_workers, 50))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for seed_id, seed_text in enumerate(seeds):
            futures.append(
                ex.submit(
                    process_seed_once,
                    client,
                    args.model_endpoint,
                    seed_id,
                    seed_text,
                    args.temperature,
                    args.max_output_tokens,
                    args.retries,
                    args.format_retries,
                    args.diversity_retries,
                    require_words8,
                    args.max_after_toks,
                    args.max_sent_sim,
                    args.min_unique_sent,
                    ban_sensitive,
                )
            )

        for fut in as_completed(futures):
            rec, seedm = fut.result()

            if seedm["status"] == "ok" and seedm["format_ok"]:
                ok_count += 1
            if seedm.get("latency_sec_last", 0.0) > 0:
                latencies.append(float(seedm["latency_sec_last"]))

            with lock:
                f_seedm.write(json.dumps(seedm, ensure_ascii=False) + "\n")
                f_seedm.flush()

                if (not args.only_ok) or (rec["status"] == "ok" and rec["format_ok"]):
                    f_bison.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    if rec["status"] == "ok" and rec["format_ok"] and isinstance(rec["output_text"], str):
                        gemini_line = {
                            "contents": [
                                {"role": "user", "parts": [{"text": rec["input_text"]}]},
                                {"role": "model", "parts": [{"text": rec["output_text"]}]},
                            ]
                        }
                        f_gemini.write(json.dumps(gemini_line, ensure_ascii=False) + "\n")

                    f_bison.flush()
                    f_gemini.flush()

            done += 1

            if args.report_every > 0 and done % args.report_every == 0:
                lat_sorted = sorted(latencies)
                p50 = lat_sorted[int((len(lat_sorted)-1)*0.50)] if lat_sorted else 0.0
                p90 = lat_sorted[int((len(lat_sorted)-1)*0.90)] if lat_sorted else 0.0
                ok_rate = ok_count / done if done else 0.0
                print(
                    f"\n[LOG] done={done}/{total} ok_rate={ok_rate:.4f} latency_p50={p50:.3f}s p90={p90:.3f}s",
                    file=sys.stderr,
                )

            print_progress(done, total)

    f_bison.close()
    f_gemini.close()
    f_seedm.close()

    lat_sorted = sorted(latencies)
    p50 = lat_sorted[int((len(lat_sorted)-1)*0.50)] if lat_sorted else 0.0
    p90 = lat_sorted[int((len(lat_sorted)-1)*0.90)] if lat_sorted else 0.0
    ok_rate = ok_count / total if total else 0.0

    print(
        "\n[REPORT]\n"
        f"  seeds_processed: {total}\n"
        f"  ok_rate        : {ok_rate:.6f}\n"
        f"  latency_sec p50/p90 : {p50:.3f}s / {p90:.3f}s\n"
        f"\n[INFO] outputs:\n"
        f"  Bison       : {out_bison_path}\n"
        f"  Gemini      : {out_gemini_path}\n"
        f"  Seed metrics: {seed_metrics_path}\n",
        file=sys.stderr,
    )

if __name__ == "__main__":
    main()
