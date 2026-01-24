from collections import Counter
import re

from datasets import load_dataset

from tpo_mm_policy_v5 import (
    VLLMOpenAIMMPolicyV5,
    classify_question,
    normalize_answer,
    rules_for,
)

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
TPO_STEPS = 2
TPO_TEMPERATURE = 0.9
TPO_TOP_P = 0.9
FINAL_TEMPERATURE = 0.0

MAX_QUESTIONS = 8
MAX_SCAN = 400


def vote_texts(texts, qtype):
    norm = [normalize_answer(t, qtype) for t in texts]
    voted = max(norm, key=norm.count)
    return voted, norm


def _word_count(text):
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))


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


def _score_candidate(text, qtype, freq_map, pool_size):
    norm = normalize_answer(text, qtype)
    freq = freq_map.get(norm, 0) / max(1, pool_size)
    fmt = _format_score(text, norm, qtype)
    unknown_penalty = 0.4 if norm == "Unknown" else 0.0
    return fmt + (0.6 * freq) - unknown_penalty


def _select_top_k(scored, k):
    selected = []
    seen = set()
    for score, text, norm in scored:
        if norm in seen:
            continue
        selected.append((score, text, norm))
        seen.add(norm)
        if len(selected) >= k:
            break
    if not selected and scored:
        selected = [scored[0]]
    return selected


def tpo_optimize_text(policy, question, qtype, candidates):
    pool = candidates[:]
    rules, _ = rules_for(qtype)
    for _ in range(TPO_STEPS):
        norm_pool = [normalize_answer(t, qtype) for t in pool]
        freq_map = Counter(norm_pool)
        scored = [
            (_score_candidate(t, qtype, freq_map, len(pool)), t, n)
            for t, n in zip(pool, norm_pool)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = _select_top_k(scored, TOP_K)

        new_candidates = []
        for _, text, _ in selected:
            prompt = (
                f"{rules}\n"
                f"Question: {question}\n"
                f"Candidate answer: {text}\n"
                "Improve the answer while keeping it concise and in the required format. Output only the answer."
            )
            new_candidates.extend(
                policy.generate_n_text(
                    prompt,
                    n=MUTATE_N,
                    temperature=TPO_TEMPERATURE,
                    top_p=TPO_TOP_P,
                    max_tokens=32,
                )
            )

        pool = [t for _, t, _ in selected] + new_candidates
        if len(pool) > POOL_SIZE:
            norm_pool = [normalize_answer(t, qtype) for t in pool]
            freq_map = Counter(norm_pool)
            scored = [
                (_score_candidate(t, qtype, freq_map, len(pool)), t, n)
                for t, n in zip(pool, norm_pool)
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            pool = [t for _, t, _ in scored[:POOL_SIZE]]

    voted, _ = vote_texts(pool, qtype)
    return pool, voted


def final_verify(policy, image, question, qtype, best_answer):
    rules, max_tokens = rules_for(qtype)
    prompt = (
        f"{rules}\n"
        f"Question: {question}\n"
        f"Proposed answer: {best_answer}\n"
        "Verify with the image and output only the answer."
    )
    return policy._ask_once_mm(image, prompt, temperature=FINAL_TEMPERATURE, max_tokens=max_tokens)


ds = load_dataset("mdwiratathya/SLAKE-vqa-english", split="test", streaming=True)
it = iter(ds)
policy = VLLMOpenAIMMPolicyV5("http://localhost:8000/v1", "medgemma-27b-it")

seen = 0

for qi in range(MAX_SCAN):
    ex = next(it)
    img = ex["image"]
    q = ex["question"]
    gt = ex.get("answer", None)

    qtype = classify_question(q)
    if qtype in YESNO_TYPES or qtype not in TARGET_QTYPES:
        continue
    if qtype == "short" and not _is_ambiguous_short(q):
        continue

    init_out = policy.generate_n_mm(img, q, n=INIT_N, use_gate=True, temperature=1.2, top_p=0.85)
    init_voted = init_out.voted
    init_stats = _disagreement_stats(init_out.texts, qtype)
    if init_stats["unique"] < MIN_UNIQUE or init_stats["disagree"] < MIN_DISAGREE:
        continue

    refined_texts = init_out.texts
    refined_voted = init_voted
    if qtype not in YESNO_TYPES:
        refined_texts, refined_voted = tpo_optimize_text(policy, q, qtype, init_out.texts)

    final_ans = final_verify(policy, img, q, qtype, refined_voted)

    print("=" * 70)
    print(f"[Q{qi}] type={qtype}")
    print("Q :", q)
    if gt is not None:
        print("GT:", gt)
    print("init samples:")
    for i, t in enumerate(init_out.texts):
        print(f"  {i}: {t}")
    print(
        f"init stats: unique={init_stats['unique']} "
        f"disagree={init_stats['disagree']:.2f} "
        f"unknown={init_stats['unknown_frac']:.2f}"
    )
    print("init voted:", init_voted)
    if qtype not in YESNO_TYPES:
        print("refined samples:")
        for i, t in enumerate(refined_texts):
            print(f"  {i}: {t}")
        print("refined voted:", refined_voted)
    print("final:", final_ans)

    seen += 1
    if seen >= MAX_QUESTIONS:
        break
