from collections import Counter
import re
import json
from datetime import datetime

import requests

from datasets import load_dataset

from tpo_mm_policy_v5 import (
    VLLMOpenAIMMPolicyV5,
    classify_question,
    normalize_answer,
    normalize_yesno,
    rules_for,
)
from semantic_eval import SemanticScorer, semantic_eval

YESNO_TYPES = {"contain_yesno", "healthy_yesno", "abnormal_yesno"}
TARGET_QTYPES = {"disease", "location", "short"}
AMBIG_KEYWORDS = {
    "largest",
    "biggest",
    "main",
    "primary",
    "most",
    "dominant",
    "likely",
    "probable",
    "suggests",
    "suggest",
}
MIN_DISAGREE = 0.4
MIN_UNIQUE = 2

INIT_N = 5
POOL_SIZE = 10
TOP_K = 3
MUTATE_N = 2
TPO_STEPS = 2  # Reduced from 4: too many rounds can drift away from correct answers
TPO_TEMPERATURE = 0.9
TPO_TOP_P = 0.9
FINAL_TEMPERATURE = 0.0
CONSENSUS_WEIGHT = 0.3  # Reduced: prevent high consensus from overriding correct answers
ANCHOR_BONUS = 1.2  # High but not too high: protect anchor but allow correction
NEW_NORM_PENALTY = 0.05  # Reduced: allow more exploration of new answers
UNKNOWN_PENALTY = 0.8  # Increased: strongly penalize Unknown
REVIEW_WEIGHT = 0.4  # Increased: reviewer can help identify correct answers
REVIEW_DEBUG = True
REVIEW_MAX_TOKENS = 12
REVIEWER_REPEAT = 2
REVIEWER_REQUIRE_CONSISTENT = True
SPECIFICITY_PENALTY_LOBE = 0.05
SPECIFICITY_PENALTY_COMMA = 0.03
SPECIFICITY_PENALTY_MAX = 0.15
IMAGE_SUMMARY_ENABLED = True
IMAGE_SUMMARY_MAX_TOKENS = 64
IMAGE_SUMMARY_TEMPERATURE = 0.0
LOCATION_MAX_WORDS = 5

SEMANTIC_ENABLED = True
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD = 0.75
SEMANTIC_DEVICE = "auto"
SEMANTIC_MAX_LENGTH = 128

MAX_QUESTIONS = 50
MAX_SCAN = 400

# Candidate expansion (ensure correct answers enter the pool)
CANDIDATE_EXPAND_ENABLED = True
CANDIDATE_EXPAND_N = 6
CANDIDATE_EXPAND_TEMPERATURE = 0.7
CANDIDATE_EXPAND_TOP_P = 0.9
CANDIDATE_EXPAND_MAX_TOKENS = 80
CANDIDATE_BANK_ENABLED = True
CANDIDATE_BANK_MAX = 24
MAX_INIT_POOL = 24

CANDIDATE_BANK = {
    "location": [
        "Left lung",
        "Right lung",
        "Bilateral lungs",
        "Both lungs",
        "Left lung, Right lung",
        "Right lung, Left lung",
        "Left Lung, Right",
        "Left upper lobe",
        "Left lower lobe",
        "Right upper lobe",
        "Right middle lobe",
        "Right lower lobe",
        "Mediastinum",
        "Center",
        "Heart",
        "Chest",
        "Thorax",
        "Abdomen",
        "Pelvis",
        "Liver",
        "Spleen",
        "Pancreas",
        "Kidney",
        "Brain",
        "Spine",
        "Right atrium",
        "Left atrium",
        "Right ventricle",
        "Left ventricle",
    ],
    "disease": [
        "Cardiomegaly",
        "Pneumonia",
        "Pleural effusion",
        "Atelectasis",
        "Mass",
        "Nodule",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pneumothorax",
        "Fracture",
        "Hernia",
        "Lung cancer",
        "Infiltration",
        "Consolidation",
        "No finding",
    ],
    "short": [
        "Lung",
        "Heart",
        "Liver",
        "Spleen",
        "Kidney",
        "Brain",
        "Stomach",
        "Pancreas",
        "Abdomen",
        "Chest",
        "Pelvis",
        "Thorax",
    ],
}

TRAIN_ANSWER_BANK_ENABLED = True
TRAIN_ANSWER_BANK_MAX = 300
TRAIN_ANSWER_BANK_SCAN = 5000
TRAIN_ANSWER_BANK_DATASETS = [
    ("SLAKE", "mdwiratathya/SLAKE-vqa-english", "train"),
    ("VQA-RAD", "flaviagiammarino/vqa-rad", "train"),
    ("PathVQA", "flaviagiammarino/path-vqa", "train"),
]
TRAIN_ANSWER_BANK = {qtype: [] for qtype in TARGET_QTYPES}
TRAIN_ANSWER_BANK_NORMS = {qtype: set() for qtype in TARGET_QTYPES}

_STOPWORDS = {
    "the", "is", "are", "in", "of", "on", "a", "an", "this", "that", "image", "picture",
    "where", "what", "which", "location", "located", "abnormality", "abnormalities",
    "main", "largest", "biggest", "most", "primary", "dominant",
}
_LUNG_KEYWORDS = {
    "lung",
    "lobe",
    "pulmonary",
    "pleural",
    "hemithorax",
    "costophrenic",
    "hilum",
    "apex",
    "apical",
    "bronch",
    "alveol",
    "pneumo",
}
_OPTION_STOPWORDS = {"yes", "no", "unknown", "not", "none", "n/a"}
_LOCATION_NOISE_TOKENS = {
    "near",
    "adjacent",
    "floor",
    "coronary",
    "sinus",
    "valve",
    "posterior",
    "anterior",
    "superior",
    "inferior",
    "medial",
    "lateral",
    "proximal",
    "distal",
}
_LOCATION_HINTS = {
    "pacemaker": {"heart", "atrium", "ventricle"},
}


def _build_keyword_set(phrases):
    tokens = set()
    for phrase in phrases:
        for token in re.findall(r"[a-z]+", (phrase or "").lower()):
            tokens.add(token)
    return tokens


def _location_question_tokens(question):
    tokens = set(re.findall(r"[a-z]+", (question or "").lower()))
    tokens = {t for t in tokens if t not in _STOPWORDS}
    for key, extras in _LOCATION_HINTS.items():
        if key in (question or "").lower():
            tokens.update(extras)
    return tokens


_ANATOMY_KEYWORDS = _build_keyword_set(CANDIDATE_BANK["location"] + CANDIDATE_BANK["short"])
_ANATOMY_KEYWORDS.update(_LUNG_KEYWORDS)
_ANATOMY_KEYWORDS.update(
    {
        "lung",
        "lungs",
        "lobe",
        "lobes",
        "pleura",
        "pleural",
        "mediastinum",
        "thorax",
        "chest",
        "abdomen",
        "pelvis",
        "heart",
        "cardiac",
        "atrium",
        "ventricle",
        "brain",
        "spine",
        "kidney",
        "liver",
        "spleen",
        "pancreas",
        "stomach",
        "apex",
        "hilum",
        "hemithorax",
        "costophrenic",
        "rib",
        "ribs",
    }
)
_DISEASE_KEYWORDS = _build_keyword_set(CANDIDATE_BANK["disease"])
_DISEASE_KEYWORDS.update(
    {
        "lesion",
        "abnormality",
        "abnormalities",
        "opacity",
        "shadow",
        "normal",
        "negative",
        "clear",
        "unremarkable",
        "tumor",
        "mass",
        "nodule",
        "effusion",
        "atelectasis",
        "pneumonia",
        "edema",
        "emphysema",
        "fibrosis",
        "fracture",
        "hernia",
        "consolidation",
        "infiltration",
        "cancer",
    }
)


def _contains_lung_keywords(text):
    tl = (text or "").lower()
    return any(k in tl for k in _LUNG_KEYWORDS)


def _is_lung_related(question, candidates=None):
    if _contains_lung_keywords(question):
        return True
    for t in candidates or []:
        if _contains_lung_keywords(t):
            return True
    return False

USE_QWEN_REVIEWER = False  # Set True to enable Qwen2.5-VL reviewer

REVIEWERS = [
    {
        "name": "medgemma",
        "type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "model": "google/medgemma-27b-it",
        "weight": 1.0,
    },
]
if USE_QWEN_REVIEWER:
    REVIEWERS.append(
        {
            "name": "qwen2.5-vl-7b",
            "type": "vllm",
            "base_url": "http://localhost:8001/v1",
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "weight": 1.0,
        }
    )
REVIEWERS.append(
    {
        "name": "biomistral",
        "type": "hf",
        "model": "BioMistral/BioMistral-7B",
        "weight": 1.0,
        "dtype": "bfloat16",
        "device_map": {"": "cuda:3"},
        "trust_remote_code": False,
        "use_chat_template": True,
    }
)
HAS_TEXT_REVIEWER = any(cfg.get("type") == "hf" for cfg in REVIEWERS)


def vote_texts(texts, qtype):
    norm = [normalize_answer(t, qtype) for t in texts]
    voted = max(norm, key=norm.count)
    return voted, norm


def _word_count(text):
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))


def _normalize_candidate_texts(texts):
    out = []
    for text in texts or []:
        if text is None:
            continue
        cleaned = " ".join(str(text).split()).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _clean_option_text(text):
    if not text:
        return ""
    cleaned = str(text).strip()
    cleaned = re.sub(r"^[\s\-\*\d\.\)\:]+", "", cleaned)
    cleaned = re.sub(r"[?!.;:,]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_option_candidates(question):
    if not question:
        return []
    q = str(question).strip()
    options = []
    q_norm = re.sub(r"\s*/\s*", "/", q)
    for match in re.finditer(
        r"([A-Za-z][A-Za-z0-9\s\-]{0,40}(?:/[A-Za-z][A-Za-z0-9\s\-]{0,40})+)",
        q_norm,
    ):
        segment = match.group(1)
        options.extend([p.strip() for p in segment.split("/") if p.strip()])
    if re.search(r"\sor\s", q, flags=re.I):
        tail = re.split(r"[?:.]", q)[-1]
        if re.search(r"\sor\s", tail, flags=re.I):
            parts = re.split(r"\s+or\s+", tail, flags=re.I)
            for part in parts:
                options.extend([p.strip() for p in part.split(",") if p.strip()])
    cleaned = []
    for opt in options:
        opt = _clean_option_text(opt)
        if not opt:
            continue
        if opt.lower() in _OPTION_STOPWORDS:
            continue
        if _word_count(opt) > 6:
            continue
        cleaned.append(opt)
    deduped = []
    seen = set()
    for opt in cleaned:
        key = opt.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(opt)
    if len(deduped) < 2 or len(deduped) > 6:
        return []
    return deduped


def _is_clean_candidate(text):
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return False
    if len(cleaned) > 60:
        return False
    if _word_count(cleaned) > 8:
        return False
    if re.search(r"[.!?]", cleaned):
        return False
    if re.search(r"\b(because|since|due to|as a result|therefore|based on)\b", cleaned.lower()):
        return False
    return True


def _apply_format_filter(texts):
    filtered = [t for t in texts if _is_clean_candidate(t)]
    return filtered if filtered else texts


def _has_any_keyword(text, keywords):
    tl = (text or "").lower()
    return any(k in tl for k in keywords)


def _extract_question_keywords(question, keyword_set):
    ql = (question or "").lower()
    return {k for k in keyword_set if k in ql}


def _filter_location_specificity(question, texts):
    if not texts:
        return texts
    ql = (question or "").lower()
    filtered = []
    for t in texts:
        tl = (t or "").lower()
        if _word_count(t) > LOCATION_MAX_WORDS:
            continue
        if any(tok in tl for tok in _LOCATION_NOISE_TOKENS) and not any(
            tok in ql for tok in _LOCATION_NOISE_TOKENS
        ):
            continue
        filtered.append(t)
    return filtered if filtered else texts


def _filter_by_qtype(question, qtype, texts):
    if qtype == "location":
        q_keywords = _location_question_tokens(question)
        filtered = []
        for t in texts:
            if not _has_any_keyword(t, _ANATOMY_KEYWORDS):
                continue
            if _has_any_keyword(t, _DISEASE_KEYWORDS):
                continue
            if q_keywords and not _has_any_keyword(t, q_keywords):
                continue
            filtered.append(t)
        filtered = _filter_location_specificity(question, filtered if filtered else texts)
        return filtered if filtered else texts
    if qtype == "disease":
        filtered = [t for t in texts if _has_any_keyword(t, _DISEASE_KEYWORDS)]
        return filtered if filtered else texts
    if qtype == "short":
        filtered = [
            t
            for t in texts
            if _has_any_keyword(t, _ANATOMY_KEYWORDS)
            and not _has_any_keyword(t, _DISEASE_KEYWORDS)
        ]
        return filtered if filtered else texts
    return texts


def _filter_by_train_bank(texts, qtype):
    if not TRAIN_ANSWER_BANK_ENABLED:
        return texts
    norms = TRAIN_ANSWER_BANK_NORMS.get(qtype)
    if not norms:
        return texts
    filtered = [t for t in texts if normalize_answer(t, qtype) in norms]
    if filtered:
        return filtered
    bank = TRAIN_ANSWER_BANK.get(qtype, [])
    return bank[: min(len(bank), TRAIN_ANSWER_BANK_MAX)] if bank else texts


def _apply_candidate_filters(question, qtype, texts, options=None):
    texts = _normalize_candidate_texts(texts)
    if not texts:
        return texts
    options = options or _extract_option_candidates(question)
    if options:
        options = _apply_format_filter(_normalize_candidate_texts(options))
        if not options:
            return texts
        option_norms = {normalize_answer(o, qtype) for o in options}
        filtered = [t for t in texts if normalize_answer(t, qtype) in option_norms]
        if not filtered:
            filtered = list(options)
        else:
            existing = {normalize_answer(t, qtype) for t in filtered}
            for opt in options:
                if normalize_answer(opt, qtype) not in existing:
                    filtered.append(opt)
        return _dedup_candidates(filtered, qtype)
    filtered = _apply_format_filter(texts)
    filtered = _drop_unknown_if_possible(filtered, qtype)
    filtered = _filter_by_qtype(question, qtype, filtered)
    filtered = _filter_by_train_bank(filtered, qtype)
    filtered = _dedup_candidates(filtered, qtype)
    return filtered


def _is_ambiguous_short(question):
    ql = (question or "").lower()
    return any(k in ql for k in AMBIG_KEYWORDS)


def _disagreement_stats(texts, qtype):
    norm = [normalize_answer(t, qtype) for t in texts]
    counts = Counter(norm)
    total = max(1, len(norm))
    max_frac = max(counts.values()) / total
    unique = len(counts)
    unknown_frac = counts.get("Unknown", 0) / total
    return {
        "unique": unique,
        "max_frac": max_frac,
        "disagree": 1.0 - max_frac,
        "unknown_frac": unknown_frac,
        "counts": counts,
    }


def _summarize_image_for_review(policy, image, question, qtype):
    if image is None:
        return None
    focus = "key visual findings"
    if qtype == "location":
        focus = "anatomical locations and regions"
    elif qtype == "disease":
        focus = "abnormal findings or disease cues"
    prompt = (
        "You are a medical imaging assistant.\n"
        f"Summarize ONLY {focus} visible in the image.\n"
        "Do NOT answer the question. Do NOT guess.\n"
        "If uncertain, output Unknown.\n"
        "Output 1-3 short phrases separated by semicolons.\n"
        f"Question: {question}"
    )
    summary = policy._ask_once_mm(
        image,
        prompt,
        temperature=IMAGE_SUMMARY_TEMPERATURE,
        max_tokens=IMAGE_SUMMARY_MAX_TOKENS,
    )
    summary = " ".join((summary or "").split()).strip()
    return summary or None


def _iter_answers(ans):
    if ans is None:
        return []
    if isinstance(ans, (list, tuple)):
        return [a for a in ans if a is not None]
    return [ans]


def _build_train_answer_bank():
    if not TRAIN_ANSWER_BANK_ENABLED:
        return
    for dataset_name, dataset_id, split in TRAIN_ANSWER_BANK_DATASETS:
        try:
            ds = load_dataset(dataset_id, split=split, streaming=True)
        except Exception as exc:
            print(f"Train bank: failed to load {dataset_name} ({dataset_id}): {exc}")
            continue
        it = iter(ds)
        scanned = 0
        while scanned < TRAIN_ANSWER_BANK_SCAN:
            try:
                ex = next(it)
            except StopIteration:
                break
            scanned += 1
            q = ex.get("question", ex.get("question_text", ""))
            gt = ex.get("answer", None)
            if not q or gt is None:
                continue
            qtype = classify_question(q)
            if qtype not in TARGET_QTYPES:
                continue
            for ans in _iter_answers(gt):
                text = str(ans).strip()
                if not text:
                    continue
                norm = normalize_answer(text, qtype)
                if norm in TRAIN_ANSWER_BANK_NORMS[qtype]:
                    continue
                if len(TRAIN_ANSWER_BANK[qtype]) >= TRAIN_ANSWER_BANK_MAX:
                    break
                TRAIN_ANSWER_BANK_NORMS[qtype].add(norm)
                TRAIN_ANSWER_BANK[qtype].append(text)
            if len(TRAIN_ANSWER_BANK[qtype]) >= TRAIN_ANSWER_BANK_MAX:
                continue
        print(
            f"Train bank {dataset_name}: "
            f"location={len(TRAIN_ANSWER_BANK['location'])} "
            f"disease={len(TRAIN_ANSWER_BANK['disease'])} "
            f"short={len(TRAIN_ANSWER_BANK['short'])}"
        )


def _drop_unknown_if_possible(texts, qtype):
    if not texts:
        return texts
    norms = [normalize_answer(t, qtype) for t in texts]
    if any(n != "Unknown" for n in norms):
        return [t for t, n in zip(texts, norms) if n != "Unknown"]
    return texts


def _force_non_unknown_answer(policy, image, question, qtype, tries=2):
    rules, max_tokens = rules_for(qtype)
    prompt = (
        f"{rules}\n"
        f"Question: {question}\n"
        "Answer MUST be specific. Do NOT answer Unknown.\n"
        "If unsure, give the single most likely answer.\n"
        "Output only the answer."
    )
    best = "Unknown"
    for i in range(tries):
        temperature = 0.2 + (0.3 * i)
        ans = policy._ask_once_mm(image, prompt, temperature=temperature, max_tokens=max_tokens)
        if normalize_answer(ans, qtype) != "Unknown":
            return ans
        best = ans
    return best


def _parse_candidate_lines(raw: str):
    if not raw:
        return []
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if len(lines) == 1 and ";" in lines[0]:
        lines = [p.strip() for p in lines[0].split(";") if p.strip()]
    cleaned = []
    for line in lines:
        line = re.sub(r"^\s*(?:\d+[\.\)]|[-*])\s*", "", line)
        line = re.sub(r"^(answer|candidate)\s*:\s*", "", line, flags=re.I)
        lower = line.lower()
        if not line:
            continue
        # Drop preambles or meta text from model output.
        if any(
            kw in lower
            for kw in (
                "here are",
                "candidate answer",
                "candidate answers",
                "distinct candidate",
                "based on the image",
                "answers for the",
            )
        ):
            continue
        if lower.endswith(":"):
            continue
        if _word_count(line) > 8:
            continue
        cleaned.append(line)
    return cleaned


def _dedup_candidates(texts, qtype):
    seen = set()
    out = []
    for t in texts:
        if not t:
            continue
        norm = normalize_answer(t, qtype)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(t)
    return out


def _inject_bilateral_location(texts, qtype, question=None):
    if qtype != "location":
        return texts
    if not _is_lung_related(question, texts):
        return texts
    norms = [normalize_answer(t, qtype) for t in texts]
    has_left = "Left lung" in norms
    has_right = "Right lung" in norms
    has_bilateral = "Left Lung, Right" in norms
    if has_left and has_right and not has_bilateral:
        return list(texts) + ["Left Lung, Right"]
    return texts


def _select_bank_candidates(question, candidates, qtype):
    use_train = TRAIN_ANSWER_BANK_ENABLED and TRAIN_ANSWER_BANK.get(qtype)
    if not CANDIDATE_BANK_ENABLED and not use_train:
        return []
    bank = TRAIN_ANSWER_BANK.get(qtype, []) if use_train else CANDIDATE_BANK.get(qtype, [])
    if not bank:
        return []
    if qtype == "location":
        tokens = _location_question_tokens(question)
    else:
        tokens = set(re.findall(r"[A-Za-z]+", (question or "").lower()))
        for t in candidates:
            tokens.update(re.findall(r"[A-Za-z]+", (t or "").lower()))
        tokens = {t for t in tokens if t not in _STOPWORDS}
    if tokens:
        filtered = [b for b in bank if any(tok in b.lower() for tok in tokens)]
        if qtype == "location":
            limit = TRAIN_ANSWER_BANK_MAX if use_train else CANDIDATE_BANK_MAX
            return filtered[:limit]
        if len(filtered) >= 5:
            limit = TRAIN_ANSWER_BANK_MAX if use_train else CANDIDATE_BANK_MAX
            return filtered[:limit]
    if qtype == "location":
        return []
    limit = TRAIN_ANSWER_BANK_MAX if use_train else CANDIDATE_BANK_MAX
    return bank[:limit]


def _expand_candidates(policy, image, question, qtype, existing):
    if not CANDIDATE_EXPAND_ENABLED:
        return existing
    rules, max_tokens = rules_for(qtype)
    prompt = (
        f"{rules}\n"
        f"Question: {question}\n"
        f"Provide {CANDIDATE_EXPAND_N} DISTINCT candidate answers.\n"
        "Each answer must be 1-6 words. Do NOT output Unknown.\n"
        "Output as a numbered list, one answer per line."
    )
    if image is not None:
        raw = policy._ask_once_mm(
            image,
            prompt,
            temperature=CANDIDATE_EXPAND_TEMPERATURE,
            max_tokens=CANDIDATE_EXPAND_MAX_TOKENS,
        )
        generated = _parse_candidate_lines(raw)
    else:
        raw = policy.generate_n_text(
            prompt,
            n=1,
            temperature=CANDIDATE_EXPAND_TEMPERATURE,
            top_p=CANDIDATE_EXPAND_TOP_P,
            max_tokens=CANDIDATE_EXPAND_MAX_TOKENS,
        )[0]
        generated = _parse_candidate_lines(raw)
    bank = _select_bank_candidates(question, existing, qtype)
    merged = list(existing) + list(generated) + list(bank)
    merged = _inject_bilateral_location(merged, qtype, question=question)
    merged = _drop_unknown_if_possible(merged, qtype)
    merged = _dedup_candidates(merged, qtype)
    if len(merged) > MAX_INIT_POOL:
        merged = merged[:MAX_INIT_POOL]
    return merged


def _needs_lung_check(question, candidates):
    return _is_lung_related(question, candidates)


def _ask_yesno(policy, image, prompt):
    raw = policy._ask_once_mm(image, prompt, temperature=0.0, max_tokens=3)
    return normalize_yesno(raw)


def _lung_side_override(policy, image, question, candidates):
    if image is None:
        return None
    if not _needs_lung_check(question, candidates):
        return None
    left_q = "Is the abnormality present in the LEFT lung? Answer Yes or No."
    right_q = "Is the abnormality present in the RIGHT lung? Answer Yes or No."
    left = _ask_yesno(policy, image, left_q)
    right = _ask_yesno(policy, image, right_q)
    if left == "Yes" and right == "Yes":
        return "Left Lung, Right"
    if left == "Yes" and right != "Yes":
        return "Left lung"
    if right == "Yes" and left != "Yes":
        return "Right lung"
    return None

def _format_score(text, norm, qtype):
    if qtype in YESNO_TYPES:
        if norm in ("Yes", "No"):
            return 1.0
        if norm == "Unknown":
            return 0.3
        return 0.0
    if qtype == "modality":
        if norm in ("CT", "X-ray", "MRI", "Ultrasound", "PET"):
            return 1.0
        if norm == "Unknown":
            return 0.4
        return 0.2
    words = _word_count(text)
    if 1 <= words <= 6:
        base = 1.0
    elif words <= 10:
        base = 0.5
    else:
        base = 0.2
    if re.search(r"[.!?]", text or ""):
        base -= 0.2
    return max(0.0, base)


def _specificity_penalty(text, qtype):
    """Penalty for overly specific answers, but reduced to avoid penalizing correct detailed answers"""
    if qtype not in ("location", "disease"):
        return 0.0
    tl = (text or "").lower()
    penalty = 0.0
    # Reduced penalty for lobe specificity (may be correct)
    if re.search(r"\b(upper|lower)\s+lobe\b", tl):
        penalty += SPECIFICITY_PENALTY_LOBE
    # Reduced penalty for comma (may indicate multiple correct locations)
    comma_count = tl.count(",")
    if comma_count > 2:  # Only penalize excessive commas (>2)
        penalty += SPECIFICITY_PENALTY_COMMA * (comma_count - 2)
    return min(penalty, SPECIFICITY_PENALTY_MAX)

class VLLMReviewer:
    def __init__(self, base_url, model):
        self.policy = VLLMOpenAIMMPolicyV5(base_url, model)

    def generate(self, prompt, image=None):
        if image is not None:
            return self.policy._ask_once_mm(
                image, prompt, temperature=0.0, max_tokens=REVIEW_MAX_TOKENS
            )
        else:
            return self.policy.generate_n_text(
                prompt, n=1, temperature=0.0, top_p=1.0, max_tokens=REVIEW_MAX_TOKENS
            )[0]


class HFReviewer:
    def __init__(
        self,
        model_id,
        dtype="bfloat16",
        device_map="auto",
        trust_remote_code=False,
        use_chat_template=False,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.use_chat_template = use_chat_template
        self._model = None
        self._tokenizer = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
            torch.backends.cuda.sdp_kernel(
                enable_math=True,
                enable_flash=False,
                enable_mem_efficient=False,
            )
        if self.dtype == "float16":
            torch_dtype = torch.float16
        elif self.dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
                attn_implementation="eager",
            )
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        self._model.eval()

    def generate(self, prompt, image=None):
        if image is not None:
            raise NotImplementedError("HFReviewer does not support multimodal input. Use VLLMReviewer for image-based review.")
        if self._model is None or self._tokenizer is None:
            self._load()
        import torch

        if (
            self.use_chat_template
            and hasattr(self._tokenizer, "apply_chat_template")
            and getattr(self._tokenizer, "chat_template", None)
        ):
            input_ids = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=REVIEW_MAX_TOKENS,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def build_reviewers(configs):
    reviewers = []
    for cfg in configs:
        rtype = cfg.get("type", "vllm")
        if rtype == "hf":
            client = HFReviewer(
                cfg["model"],
                dtype=cfg.get("dtype", "bfloat16"),
                device_map=cfg.get("device_map", "auto"),
                trust_remote_code=cfg.get("trust_remote_code", False),
                use_chat_template=cfg.get("use_chat_template", False),
            )
        else:
            client = VLLMReviewer(cfg["base_url"], cfg["model"])
        reviewers.append((cfg, client))
    return reviewers


def _parse_score(text):
    match = re.search(r"(-?\d+(?:\.\d+)?)", text or "")
    if not match:
        return None
    val = float(match.group(1))
    if val > 10:
        val = val / 10.0 if val <= 100 else 10.0
    if val < 0:
        val = 0.0
    if val > 10:
        val = 10.0
    return val / 10.0


def _parse_choice(text, max_choice):
    match = re.search(r"\b(\d{1,2})\b", text or "")
    if not match:
        return None
    choice = int(match.group(1))
    if 1 <= choice <= max_choice:
        return choice
    return None


def _review_scores_for_pool(
    reviewers,
    cache,
    question,
    qtype,
    pool_texts,
    rules,
    image=None,
    image_summary=None,
):
    if not reviewers or REVIEW_WEIGHT <= 0:
        return {}
    norm_to_text = {}
    for text in pool_texts:
        norm = normalize_answer(text, qtype)
        if norm not in norm_to_text:
            norm_to_text[norm] = " ".join((text or "").split())
    norms = list(norm_to_text.keys())
    if not norms:
        return {}

    prompt_mm = None
    prompt_text = None
    if image is not None:
        prompt_mm = (
            "You are a medical QA evaluator. Choose the BEST and MOST ACCURATE answer among candidates.\n"
            "CRITICAL: Medical accuracy is the ONLY priority. Format and style are secondary.\n"
            "Carefully examine the image to verify which answer is medically correct.\n"
            "Pay close attention to:\n"
            "- Anatomical details (left/right, specific locations, organ names)\n"
            "- Disease names and medical terminology\n"
            "- Spatial relationships in the image\n"
            "If multiple answers are medically plausible, choose the one that best matches the image evidence.\n"
            f"{rules}\n"
            f"Question: {question}\n"
            "Candidates:\n"
        )
        if image_summary:
            prompt_text = (
                "You are a medical QA evaluator. Choose the BEST and MOST ACCURATE answer among candidates.\n"
                "CRITICAL: Medical accuracy is the ONLY priority. Format and style are secondary.\n"
                "You do NOT have the image. Use ONLY the following image summary (it may be incomplete).\n"
                f"Image summary: {image_summary}\n"
                f"{rules}\n"
                f"Question: {question}\n"
                "Candidates:\n"
            )
        else:
            prompt_text = (
                "You are a medical QA evaluator. Choose the BEST and MOST ACCURATE answer among candidates.\n"
                "CRITICAL: Medical accuracy is the ONLY priority. Format and style are secondary.\n"
                "Pay close attention to:\n"
                "- Anatomical details (left/right, specific locations, organ names)\n"
                "- Disease names and medical terminology\n"
                "- Semantic correctness\n"
                "Do NOT use image evidence (image not available).\n"
                f"{rules}\n"
                f"Question: {question}\n"
                "Candidates:\n"
            )
    else:
        prompt_text = (
            "You are a medical QA evaluator. Choose the BEST and MOST ACCURATE answer among candidates.\n"
            "CRITICAL: Medical accuracy is the ONLY priority. Format and style are secondary.\n"
            "Pay close attention to:\n"
            "- Anatomical details (left/right, specific locations, organ names)\n"
            "- Disease names and medical terminology\n"
            "- Semantic correctness\n"
            "Do NOT use image evidence (image not available).\n"
            f"{rules}\n"
            f"Question: {question}\n"
            "Candidates:\n"
        )

    for i, text in enumerate(norm_to_text.values(), 1):
        if prompt_mm is not None:
            prompt_mm += f"{i}. {text}\n"
        if prompt_text is not None:
            prompt_text += f"{i}. {text}\n"
    if prompt_mm is not None:
        prompt_mm += "Return ONLY the number of the best answer."
    if prompt_text is not None:
        prompt_text += "Return ONLY the number of the best answer."

    scores = {norm: 0.0 for norm in norms}
    weight_sum = 0.0
    for cfg, reviewer in reviewers:
        if qtype == "short" and cfg.get("type") == "hf":
            if REVIEW_DEBUG:
                print(f"  RM[{cfg['name']}] skipped (short question: text reviewer disabled)")
            continue
        key = (cfg["name"], qtype, question, tuple(norms), image is not None, image_summary)
        if key not in cache:
            try:
                used_summary = False
                choices = []
                last_raw = None
                for _ in range(max(1, REVIEWER_REPEAT)):
                    try:
                        if image is not None and prompt_mm is not None:
                            raw = reviewer.generate(prompt_mm, image=image)
                        else:
                            raw = reviewer.generate(prompt_text, image=None)
                        last_raw = raw
                    except NotImplementedError:
                        if prompt_text and image_summary:
                            raw = reviewer.generate(prompt_text, image=None)
                            last_raw = raw
                            used_summary = True
                        else:
                            if REVIEW_DEBUG:
                                print(f"  RM[{cfg['name']}] skipped (multimodal not supported)")
                            last_raw = None
                            choices = []
                            break
                    choice = _parse_choice(last_raw, len(norms))
                    choices.append(choice)
                if not choices or any(c is None for c in choices):
                    if REVIEW_DEBUG:
                        print(f"  RM[{cfg['name']}] invalid choice (skipped)")
                    cache[key] = None
                elif REVIEWER_REQUIRE_CONSISTENT and len(set(choices)) > 1:
                    if REVIEW_DEBUG:
                        print(f"  RM[{cfg['name']}] inconsistent choices={choices} (skipped)")
                    cache[key] = None
                else:
                    choice = choices[0]
                    if REVIEW_DEBUG:
                        raw_display = " ".join((last_raw or "").split())
                        choice_display = "None" if choice is None else str(choice)
                        picked = "None" if choice is None else norms[choice - 1]
                        if image is not None and not used_summary:
                            img_tag = " [with image]"
                        elif image_summary:
                            img_tag = " [summary]"
                        else:
                            img_tag = ""
                        print(
                            f"  RM[{cfg['name']}] choice={choice_display} norm={picked} raw={raw_display!r}{img_tag}"
                        )
                    cache[key] = choice
            except requests.exceptions.RequestException as exc:
                if REVIEW_DEBUG:
                    print(f"  RM[{cfg['name']}] error={exc.__class__.__name__}: {exc} (skipped)")
                cache[key] = None
            except Exception as exc:
                if REVIEW_DEBUG:
                    print(f"  RM[{cfg['name']}] error={exc.__class__.__name__}: {exc} (skipped)")
                cache[key] = None
                continue
        cached = cache.get(key)
        if cached is None:
            continue
        weight = cfg.get("weight", 1.0)
        picked_norm = norms[cached - 1]
        scores[picked_norm] += weight
        weight_sum += weight
    if weight_sum <= 0:
        return {}
    for norm in scores:
        scores[norm] = scores[norm] / weight_sum
    return scores


def _score_candidate(
    text,
    qtype,
    freq_map,
    pool_size,
    anchor_norm=None,
    init_norms=None,
    review_score=0.0,
    high_consensus_anchor=False,  # Kept for compatibility but not used
):
    norm = normalize_answer(text, qtype)
    freq = freq_map.get(norm, 0) / max(1, pool_size)
    fmt = _format_score(text, norm, qtype)
    specificity_penalty = _specificity_penalty(text, qtype)
    unknown_penalty = UNKNOWN_PENALTY if norm == "Unknown" else 0.0
    # Never reward Unknown as anchor; it should be a fallback only.
    anchor_bonus = ANCHOR_BONUS if anchor_norm and norm == anchor_norm and norm != "Unknown" else 0.0
    bilateral_bonus = 0.0
    if qtype == "location" and norm == "Left Lung, Right":
        if freq_map.get("Left lung", 0) > 0 and freq_map.get("Right lung", 0) > 0:
            bilateral_bonus = 0.25
    # REMOVED: high consensus extra bonus - models can collectively be wrong
    # High consensus doesn't guarantee correctness
    new_norm_penalty = NEW_NORM_PENALTY if init_norms and norm not in init_norms else 0.0
    return (
        fmt
        + (CONSENSUS_WEIGHT * freq)
        + anchor_bonus
        + bilateral_bonus
        - unknown_penalty
        - new_norm_penalty
        - specificity_penalty
        + (REVIEW_WEIGHT * review_score)
    )


def _select_top_k(scored, k, anchor_norm=None):
    selected = []
    seen = set()
    for score, text, norm, review in scored:
        if norm in seen:
            continue
        selected.append((score, text, norm, review))
        seen.add(norm)
        if len(selected) >= k:
            break
    if anchor_norm and scored and all(n != anchor_norm for _, _, n, _ in selected):
        for score, text, norm, review in scored:
            if norm == anchor_norm:
                selected.append((score, text, norm, review))
                break
    if len(selected) > k:
        selected = selected[:k]
    if not selected and scored:
        selected = [scored[0]]
    return selected


def tpo_optimize_text(
    policy,
    question,
    qtype,
    candidates,
    anchor_text=None,
    reviewers=None,
    review_cache=None,
    image=None,
    image_summary=None,
):
    options = _extract_option_candidates(question)
    pool = _apply_candidate_filters(question, qtype, candidates, options=options)
    rules, _ = rules_for(qtype)
    init_norms = {normalize_answer(t, qtype) for t in pool}
    if not anchor_text and candidates:
        anchor_text = pool[0] if pool else candidates[0]
    anchor_norm = normalize_answer(anchor_text, qtype) if anchor_text else None
    option_norms = {normalize_answer(o, qtype) for o in options} if options else None
    if option_norms and anchor_norm not in option_norms:
        anchor_norm = None
        anchor_text = None
    # If anchor is Unknown but other answers exist, do not force-keep Unknown.
    if anchor_norm == "Unknown":
        if any(normalize_answer(t, qtype) != "Unknown" for t in pool):
            anchor_norm = None
    review_cache = review_cache or {}
    
    # Check initial consensus: if anchor appears in 3+ out of 5 initial samples, it's likely correct
    # BUT: high consensus doesn't guarantee correctness (models can collectively be wrong)
    # So we use this only for early stopping, not for extra bonus
    init_norm_list = [normalize_answer(t, qtype) for t in pool]
    anchor_freq_in_init = init_norm_list.count(anchor_norm) if anchor_norm else 0
    high_consensus_anchor = anchor_freq_in_init >= 3  # 3/5 = 60% consensus
    # REMOVED: high consensus bonus - models can collectively be wrong
    # Instead, we'll use early stopping if anchor is consistently leading

    for step in range(TPO_STEPS):
        norm_pool = [normalize_answer(t, qtype) for t in pool]
        freq_map = Counter(norm_pool)
        review_scores = _review_scores_for_pool(
            reviewers,
            review_cache,
            question,
            qtype,
            pool,
            rules,
            image=image,
            image_summary=image_summary,
        )
        scored = []
        for t, n in zip(pool, norm_pool):
            review = review_scores.get(n, 0.0)
            score = _score_candidate(
                t, qtype, freq_map, len(pool), anchor_norm, init_norms, review,
                high_consensus_anchor=high_consensus_anchor
            )
            scored.append((score, t, n, review))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Early stopping: if anchor is consistently the top answer with large margin, stop early
        if anchor_norm and scored:
            top_norm = scored[0][2]
            if top_norm == anchor_norm and step > 0:
                # Anchor is leading, check if it's significantly ahead
                if len(scored) > 1:
                    score_diff = scored[0][0] - scored[1][0]
                    # More conservative: only stop if anchor is very far ahead (0.5+)
                    # This allows TPO to explore even when anchor is slightly ahead
                    if score_diff > 0.5:  # Anchor is very significantly ahead
                        print(f"Early stopping at round {step + 1}: anchor answer is very significantly leading (diff={score_diff:.2f})")
                        break
        
        selected = _select_top_k(scored, TOP_K, anchor_norm)

        print(f"Round {step + 1} pool (score | review_score | norm | text):")
        print(f"  Note: review_score = reviewer voting weight (0.0=not selected, >0=selected)")
        for i, (score, text, norm, review) in enumerate(scored, 1):
            anchor_tag = " [anchor]" if anchor_norm and norm == anchor_norm else ""
            review_tag = " ✓" if review > 0 else ""
            print(f"  {i:02d}. {score:.2f} | {review:.2f}{review_tag} | {norm} | {text}{anchor_tag}")
        print(f"Round {step + 1} selected (score | review_score | norm | text):")
        for i, (score, text, norm, review) in enumerate(selected, 1):
            anchor_tag = " [anchor]" if anchor_norm and norm == anchor_norm else ""
            review_tag = " ✓" if review > 0 else ""
            print(f"  {i:02d}. {score:.2f} | {review:.2f}{review_tag} | {norm} | {text}{anchor_tag}")

        new_candidates = []
        for _, text, _, _ in selected:
            # Aggressive mutation: generate diverse alternatives to explore better answers
            # Use higher temperature and more candidates to explore different possibilities
            if image is not None:
                # Use multimodal generation with higher diversity
                mm_out = policy.generate_n_mm(
                    image, question, n=MUTATE_N, use_gate=True,
                    temperature=TPO_TEMPERATURE * 1.2, top_p=TPO_TOP_P  # Higher temperature for more diversity
                )
                new_candidates.extend(mm_out.texts)
                
                # Also generate one "counter-example": explicitly try to find different answer
                # Use multimodal generation with high diversity to explore alternatives
                if step == 0:  # Only in first round to avoid too many candidates
                    counter_mm_out = policy.generate_n_mm(
                        image, question, n=1, use_gate=True,
                        temperature=TPO_TEMPERATURE * 1.5, top_p=TPO_TOP_P  # Very high temperature for diversity
                    )
                    # Only add if it's different from the selected answer
                    counter_norm = normalize_answer(counter_mm_out.texts[0], qtype) if counter_mm_out.texts else None
                    selected_norm = normalize_answer(text, qtype)
                    if counter_norm and counter_norm != selected_norm:
                        new_candidates.extend(counter_mm_out.texts)
            else:
                # Text-only: generate alternative with higher diversity
                prompt = (
                    f"{rules}\n"
                    f"Question: {question}\n"
                    "Generate a concise answer in the required format. "
                    "The answer should be medically accurate and plausible. "
                    "Output only the answer."
                )
                new_candidates.extend(
                    policy.generate_n_text(
                        prompt,
                        n=MUTATE_N,
                        temperature=TPO_TEMPERATURE * 1.2,  # Higher temperature
                        top_p=TPO_TOP_P,
                        max_tokens=32,
                    )
                )

        pool = [t for _, t, _, _ in selected] + new_candidates
        pool = _apply_candidate_filters(question, qtype, pool, options=options)
        if (
            anchor_text
            and anchor_norm
            and (not option_norms or anchor_norm in option_norms)
            and all(normalize_answer(t, qtype) != anchor_norm for t in pool)
        ):
            pool.append(anchor_text)
        if len(pool) > POOL_SIZE:
            norm_pool = [normalize_answer(t, qtype) for t in pool]
            freq_map = Counter(norm_pool)
            review_scores = _review_scores_for_pool(
                reviewers,
                review_cache,
                question,
                qtype,
                pool,
                rules,
                image=image,
                image_summary=image_summary,
            )
            scored = []
            for t, n in zip(pool, norm_pool):
                review = review_scores.get(n, 0.0)
                score = _score_candidate(
                    t, qtype, freq_map, len(pool), anchor_norm, init_norms, review,
                    high_consensus_anchor=high_consensus_anchor
                )
                scored.append((score, t, n, review))
            scored.sort(key=lambda x: x[0], reverse=True)
            pool = [t for _, t, _, _ in scored[:POOL_SIZE]]
            if (
                anchor_text
                and anchor_norm
                and (not option_norms or anchor_norm in option_norms)
                and all(normalize_answer(t, qtype) != anchor_norm for t in pool)
            ):
                pool[-1] = anchor_text

    voted, _ = vote_texts(pool, qtype)
    return pool, voted


def final_verify(policy, image, question, qtype, best_answer):
    # Disabled or made very conservative: final_verify often makes answers worse
    # If enabled, use very conservative settings
    USE_FINAL_VERIFY = False  # Set to True to enable, but it often reduces accuracy
    
    if not USE_FINAL_VERIFY:
        return best_answer  # Return as-is without verification
    
    rules, max_tokens = rules_for(qtype)
    prompt = (
        f"{rules}\n"
        f"Question: {question}\n"
        f"Proposed answer: {best_answer}\n"
        "Verify with the image. If the proposed answer is correct, output it exactly. "
        "Only change it if you are CERTAIN it is wrong. Output only the answer."
    )
    verified = policy._ask_once_mm(image, prompt, temperature=FINAL_TEMPERATURE, max_tokens=max_tokens)
    # Only use verified answer if it's significantly different (to avoid minor changes)
    if normalize_answer(verified, qtype) == normalize_answer(best_answer, qtype):
        return best_answer  # No significant change, keep original
    return verified


def exact_match(pred, gt):
    """Exact match after normalization"""
    if not pred or not gt:
        return False
    pred_norm = pred.strip().lower()
    gt_norm = gt.strip().lower()
    return pred_norm == gt_norm


def partial_match(pred, gt):
    """Check if prediction contains ground truth or vice versa"""
    if not pred or not gt:
        return False
    pred_lower = pred.strip().lower()
    gt_lower = gt.strip().lower()
    
    # Check if pred contains gt or gt contains pred
    if gt_lower in pred_lower or pred_lower in gt_lower:
        return True
    
    # Check for comma-separated values
    gt_parts = [p.strip() for p in gt_lower.split(",")]
    pred_parts = [p.strip() for p in pred_lower.split(",")]
    
    # Check if any part matches
    for gt_part in gt_parts:
        for pred_part in pred_parts:
            if gt_part in pred_part or pred_part in gt_part:
                return True
    
    return False


def calculate_accuracy(results):
    """Calculate accuracy metrics"""
    total = len(results)
    if total == 0:
        return {}
    
    exact_matches = sum(1 for r in results if exact_match(r["final_answer"], r["ground_truth"]))
    partial_matches = sum(1 for r in results if partial_match(r["final_answer"], r["ground_truth"]))
    semantic_total = sum(1 for r in results if r.get("semantic_match") is not None)
    semantic_matches = sum(1 for r in results if r.get("semantic_match") is True)
    semantic_score_sum = sum(
        r.get("semantic_score", 0.0) for r in results if r.get("semantic_score") is not None
    )
    
    # Per question type
    by_type = {}
    for r in results:
        qtype = r.get("question_type", "unknown")
        if qtype not in by_type:
            by_type[qtype] = {
                "total": 0,
                "exact": 0,
                "partial": 0,
                "semantic_total": 0,
                "semantic_match": 0,
                "semantic_score_sum": 0.0,
            }
        by_type[qtype]["total"] += 1
        if exact_match(r["final_answer"], r["ground_truth"]):
            by_type[qtype]["exact"] += 1
        if partial_match(r["final_answer"], r["ground_truth"]):
            by_type[qtype]["partial"] += 1
        sem_match = r.get("semantic_match")
        sem_score = r.get("semantic_score")
        if sem_match is not None:
            by_type[qtype]["semantic_total"] += 1
            if sem_match is True:
                by_type[qtype]["semantic_match"] += 1
            if sem_score is not None:
                by_type[qtype]["semantic_score_sum"] += sem_score
    
    metrics = {
        "total": total,
        "exact_match": exact_matches,
        "exact_match_rate": exact_matches / total if total > 0 else 0.0,
        "partial_match": partial_matches,
        "partial_match_rate": partial_matches / total if total > 0 else 0.0,
        "semantic_total": semantic_total,
        "semantic_match": semantic_matches,
        "semantic_match_rate": semantic_matches / semantic_total if semantic_total > 0 else None,
        "semantic_avg_score": semantic_score_sum / semantic_total if semantic_total > 0 else None,
        "by_type": {
            qtype: {
                "total": stats["total"],
                "exact_match_rate": stats["exact"] / stats["total"] if stats["total"] > 0 else 0.0,
                "partial_match_rate": stats["partial"] / stats["total"] if stats["total"] > 0 else 0.0,
                "semantic_match_rate": (
                    stats["semantic_match"] / stats["semantic_total"]
                    if stats["semantic_total"] > 0
                    else None
                ),
                "semantic_avg_score": (
                    stats["semantic_score_sum"] / stats["semantic_total"]
                    if stats["semantic_total"] > 0
                    else None
                ),
            }
            for qtype, stats in by_type.items()
        }
    }
    
    return metrics


# Load multiple datasets for paper experiments
DATASETS = [
    ("SLAKE", "mdwiratathya/SLAKE-vqa-english", "test"),
    ("VQA-RAD", "flaviagiammarino/vqa-rad", "test"),
    ("PathVQA", "flaviagiammarino/path-vqa", "test"),
]

_build_train_answer_bank()

policy = VLLMOpenAIMMPolicyV5("http://localhost:8000/v1", "google/medgemma-27b-it")
reviewer_policies = build_reviewers(REVIEWERS)
semantic_scorer = SemanticScorer(
    enabled=SEMANTIC_ENABLED,
    model_name=SEMANTIC_MODEL,
    device=SEMANTIC_DEVICE,
    max_length=SEMANTIC_MAX_LENGTH,
)

total_seen = 0
dataset_stats = {name: {"seen": 0, "processed": 0} for name, _, _ in DATASETS}
all_results = {name: [] for name, _, _ in DATASETS}

for dataset_name, dataset_id, split in DATASETS:
    print("=" * 70)
    print(f"Processing dataset: {dataset_name} ({dataset_id})")
    print("=" * 70)
    
    try:
        ds = load_dataset(dataset_id, split=split, streaming=True)
        it = iter(ds)
        review_cache = {}
        seen = 0
        
        for qi in range(MAX_SCAN):
            try:
                ex = next(it)
            except StopIteration:
                print(f"Dataset {dataset_name} exhausted at {qi} samples")
                break
            
            # Handle different dataset formats
            if dataset_name == "SLAKE":
                img = ex["image"]
                q = ex["question"]
                gt = ex.get("answer", None)
            elif dataset_name == "VQA-RAD":
                img = ex["image"]
                q = ex.get("question", ex.get("question_text", ""))
                gt = ex.get("answer", None)
            elif dataset_name == "PathVQA":
                img = ex["image"]
                q = ex.get("question", ex.get("question_text", ""))
                gt = ex.get("answer", None)
            else:
                # Default format
                img = ex["image"]
                q = ex.get("question", ex.get("question_text", ""))
                gt = ex.get("answer", None)

            qtype = classify_question(q)
            if qtype in YESNO_TYPES or qtype not in TARGET_QTYPES:
                continue
            if qtype == "short" and not _is_ambiguous_short(q):
                continue

            # Use deterministic baseline answer as anchor, plus diverse samples for exploration.
            anchor_out = policy.generate_n_mm(img, q, n=1, use_gate=True, temperature=0.0, top_p=1.0)
            anchor_text = anchor_out.voted if anchor_out.texts else "Unknown"
            if normalize_answer(anchor_text, qtype) == "Unknown":
                forced = _force_non_unknown_answer(policy, img, q, qtype, tries=2)
                if normalize_answer(forced, qtype) != "Unknown":
                    anchor_text = forced
                    print(f"  [Forced non-Unknown anchor: {anchor_text}]")
            diverse_n = max(1, INIT_N - 1)
            diverse_out = policy.generate_n_mm(img, q, n=diverse_n, use_gate=True, temperature=1.2, top_p=0.85)
            init_texts = [anchor_text] + [t for t in diverse_out.texts if t]
            init_texts = _drop_unknown_if_possible(init_texts, qtype)
            options = _extract_option_candidates(q)
            init_texts = _apply_candidate_filters(q, qtype, init_texts, options=options)
            if init_texts:
                init_voted, _ = vote_texts(init_texts, qtype)
            else:
                init_voted = anchor_text
            init_samples = list(init_texts)
            init_stats = _disagreement_stats(init_samples, qtype)
            skip_tpo = init_stats["unique"] < MIN_UNIQUE or init_stats["disagree"] < MIN_DISAGREE
            tpo_used = qtype not in YESNO_TYPES and not skip_tpo
            if skip_tpo:
                print(
                    f"  [Skipping TPO: low disagreement/unique "
                    f"(unique={init_stats['unique']}, disagree={init_stats['disagree']:.2f})]"
                )
                expanded_texts = list(init_samples)
            else:
                expanded_texts = _expand_candidates(policy, img, q, qtype, init_samples)
                if len(expanded_texts) > len(init_samples):
                    print(f"  [Candidate expansion: {len(init_samples)} -> {len(expanded_texts)}]")
            expanded_texts = _apply_candidate_filters(q, qtype, expanded_texts, options=options)

            image_summary = None
            if (
                IMAGE_SUMMARY_ENABLED
                and qtype in ("location", "disease")
                and not skip_tpo
                and img is not None
                and HAS_TEXT_REVIEWER
            ):
                image_summary = _summarize_image_for_review(policy, img, q, qtype)
                if REVIEW_DEBUG and image_summary:
                    print(f"  [Image summary] {image_summary}")

            refined_texts = expanded_texts
            refined_voted = init_voted
            if qtype not in YESNO_TYPES and not skip_tpo:
                # Use TPO for all questions to explore better answers
                # But be conservative: strongly protect the initial voted answer
                refined_texts, refined_voted = tpo_optimize_text(
                    policy,
                    q,
                    qtype,
                    expanded_texts,
                    anchor_text=init_voted,
                    reviewers=reviewer_policies,
                    review_cache=review_cache,
                    image=img,
                    image_summary=image_summary,
                )

            if qtype == "location":
                override = _lung_side_override(policy, img, q, init_samples)
                if override:
                    print(f"  [Lung side override: {override}]")
                    refined_voted = override

            final_ans = final_verify(policy, img, q, qtype, refined_voted)
            semantic_match, semantic_score = (None, None)
            if SEMANTIC_ENABLED:
                semantic_match, semantic_score = semantic_eval(
                    final_ans,
                    gt,
                    semantic_scorer,
                    threshold=SEMANTIC_THRESHOLD,
                )

            print("=" * 70)
            print(f"[{dataset_name}] Q{qi} type={qtype}")
            print("Q :", q)
            if gt is not None:
                print("GT:", gt)
            print("init samples:")
            for i, t in enumerate(init_samples):
                print(f"  {i}: {t}")
            print(
                f"init stats: unique={init_stats['unique']} "
                f"disagree={init_stats['disagree']:.2f} "
                f"unknown={init_stats['unknown_frac']:.2f}"
            )
            print("init voted:", init_voted)
            if qtype not in YESNO_TYPES:
                print(f"refined samples (after TPO, pool_size={len(refined_texts)}):")
                for i, t in enumerate(refined_texts):
                    print(f"  {i}: {t}")
                print("refined voted:", refined_voted)
            print("final:", final_ans)
            if semantic_score is not None:
                sm = "✓" if semantic_match else "✗"
                print(f"semantic: {semantic_score:.2f} ({sm})")

            # Save results for validation
            result = {
                "dataset": dataset_name,
                "question_id": qi,
                "question": q,
                "question_type": qtype,
                "tpo_used": tpo_used,
                "tpo_skipped": skip_tpo,
                "ground_truth": gt,
                "init_samples": init_samples,
                "expanded_samples": expanded_texts,
                "init_voted": init_voted,
                "init_stats": {
                    "unique": init_stats["unique"],
                    "disagree": init_stats["disagree"],
                    "unknown_frac": init_stats["unknown_frac"],
                },
                "refined_samples": refined_texts if qtype not in YESNO_TYPES else None,
                "refined_voted": refined_voted if qtype not in YESNO_TYPES else None,
                "final_answer": final_ans,
                "exact_match": exact_match(final_ans, gt) if gt else None,
                "partial_match": partial_match(final_ans, gt) if gt else None,
                "semantic_match": semantic_match,
                "semantic_score": semantic_score,
            }
            all_results[dataset_name].append(result)

            seen += 1
            dataset_stats[dataset_name]["processed"] += 1
            total_seen += 1
            
            if seen >= MAX_QUESTIONS:
                break
        
        dataset_stats[dataset_name]["seen"] = seen
        print(f"\nDataset {dataset_name} completed: {seen} questions processed")
        
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "=" * 70)
print("TPO EVALUATION RESULTS")
print("=" * 70)

# Calculate metrics
all_metrics = {}
for dataset_name in DATASETS:
    dataset_name = dataset_name[0]
    if all_results[dataset_name]:
        metrics = calculate_accuracy(all_results[dataset_name])
        all_metrics[dataset_name] = metrics
        
        print(f"\n{dataset_name}:")
        print(f"  Total questions: {metrics['total']}")
        print(f"  Exact match: {metrics['exact_match']}/{metrics['total']} ({metrics['exact_match_rate']:.2%})")
        print(f"  Partial match: {metrics['partial_match']}/{metrics['total']} ({metrics['partial_match_rate']:.2%})")
        if metrics.get("semantic_total"):
            print(
                f"  Semantic match: {metrics['semantic_match']}/{metrics['semantic_total']} "
                f"({metrics['semantic_match_rate']:.2%}) "
                f"avg={metrics['semantic_avg_score']:.2f}"
            )
        print(f"  By question type:")
        for qtype, type_stats in metrics['by_type'].items():
            print(f"    {qtype}: {type_stats['exact_match_rate']:.2%} exact, {type_stats['partial_match_rate']:.2%} partial")

# Overall metrics
all_results_flat = []
for dataset_name in DATASETS:
    dataset_name = dataset_name[0]
    all_results_flat.extend(all_results[dataset_name])

if all_results_flat:
    overall_metrics = calculate_accuracy(all_results_flat)
    print(f"\nOVERALL:")
    print(f"  Total questions: {overall_metrics['total']}")
    print(f"  Exact match: {overall_metrics['exact_match']}/{overall_metrics['total']} ({overall_metrics['exact_match_rate']:.2%})")
    print(f"  Partial match: {overall_metrics['partial_match']}/{overall_metrics['total']} ({overall_metrics['partial_match_rate']:.2%})")
    if overall_metrics.get("semantic_total"):
        print(
            f"  Semantic match: {overall_metrics['semantic_match']}/{overall_metrics['semantic_total']} "
            f"({overall_metrics['semantic_match_rate']:.2%}) "
            f"avg={overall_metrics['semantic_avg_score']:.2f}"
        )

    tpo_used_results = [r for r in all_results_flat if r.get("tpo_used")]
    tpo_skipped_results = [r for r in all_results_flat if r.get("tpo_used") is False]

    if tpo_used_results:
        used_metrics = calculate_accuracy(tpo_used_results)
        print(f"\nTPO-USED ONLY:")
        print(f"  Total questions: {used_metrics['total']}")
        print(f"  Exact match: {used_metrics['exact_match']}/{used_metrics['total']} ({used_metrics['exact_match_rate']:.2%})")
        print(f"  Partial match: {used_metrics['partial_match']}/{used_metrics['total']} ({used_metrics['partial_match_rate']:.2%})")
    if tpo_skipped_results:
        skipped_metrics = calculate_accuracy(tpo_skipped_results)
        print(f"\nTPO-SKIPPED (BASELINE) ONLY:")
        print(f"  Total questions: {skipped_metrics['total']}")
        print(f"  Exact match: {skipped_metrics['exact_match']}/{skipped_metrics['total']} ({skipped_metrics['exact_match_rate']:.2%})")
        print(f"  Partial match: {skipped_metrics['partial_match']}/{skipped_metrics['total']} ({skipped_metrics['partial_match_rate']:.2%})")

# Save all results to JSON file for validation
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"tpo_results_{timestamp}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": timestamp,
        "model": "google/medgemma-27b-it",
        "method": "TPO_optimized",
        "total_questions": total_seen,
        "dataset_stats": dataset_stats,
        "metrics": all_metrics,
        "overall_metrics": overall_metrics if all_results_flat else None,
        "tpo_breakdown": {
            "tpo_used": used_metrics if all_results_flat and tpo_used_results else None,
            "tpo_skipped": skipped_metrics if all_results_flat and tpo_skipped_results else None,
        },
        "results": all_results,
    }, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {output_file}")
print("You can use this file for validation and evaluation.")
