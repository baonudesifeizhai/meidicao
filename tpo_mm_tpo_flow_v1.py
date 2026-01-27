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
TPO_TEMPERATURE = 1.1
TPO_TOP_P = 0.85
FINAL_TEMPERATURE = 0.0
CONSENSUS_WEIGHT = 0.6
ANCHOR_BONUS = 0.3
NEW_NORM_PENALTY = 0.15
UNKNOWN_PENALTY = 0.6
REVIEW_WEIGHT = 0.8
REVIEW_DEBUG = True
REVIEW_MAX_TOKENS = 12
SPECIFICITY_PENALTY_LOBE = 0.15
SPECIFICITY_PENALTY_COMMA = 0.08
SPECIFICITY_PENALTY_MAX = 0.3

MAX_QUESTIONS = 8
MAX_SCAN = 400

REVIEWERS = [
    {
        "name": "medgemma",
        "type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "model": "google/medgemma-27b-it",
        "weight": 1.0,
    },
    {
        "name": "qwen2.5-14b",
        "type": "vllm",
        "base_url": "http://localhost:8001/v1",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "weight": 1.0,
    },
    {
        "name": "biomistral",
        "type": "hf",
        "model": "BioMistral/BioMistral-7B",
        "weight": 1.0,
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": False,
        "use_chat_template": True,
    },
]


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


def _specificity_penalty(text, qtype):
    if qtype not in ("location", "disease"):
        return 0.0
    tl = (text or "").lower()
    penalty = 0.0
    if re.search(r"\b(upper|lower)\s+lobe\b", tl):
        penalty += SPECIFICITY_PENALTY_LOBE
    comma_count = tl.count(",")
    if comma_count:
        penalty += SPECIFICITY_PENALTY_COMMA * comma_count
    return min(penalty, SPECIFICITY_PENALTY_MAX)

class VLLMReviewer:
    def __init__(self, base_url, model):
        self.policy = VLLMOpenAIMMPolicyV5(base_url, model)

    def generate(self, prompt):
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

    def generate(self, prompt):
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


def _review_scores_for_pool(reviewers, cache, question, qtype, pool_texts, rules):
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
    prompt = (
        "You are a medical QA evaluator. Choose the best answer among candidates.\n"
        "Consider format compliance and semantic relevance. Do NOT use image evidence.\n"
        f"{rules}\n"
        f"Question: {question}\n"
        "Candidates:\n"
    )
    for i, text in enumerate(norm_to_text.values(), 1):
        prompt += f"{i}. {text}\n"
    prompt += "Return ONLY the number of the best answer."
    scores = {norm: 0.0 for norm in norms}
    weight_sum = 0.0
    for cfg, reviewer in reviewers:
        key = (cfg["name"], qtype, question, tuple(norms))
        if key not in cache:
            raw = reviewer.generate(prompt)
            choice = _parse_choice(raw, len(norms))
            if REVIEW_DEBUG:
                raw_display = " ".join((raw or "").split())
                choice_display = "None" if choice is None else str(choice)
                picked = "None" if choice is None else norms[choice - 1]
                print(
                    f"  RM[{cfg['name']}] choice={choice_display} norm={picked} raw={raw_display!r}"
                )
            cache[key] = choice
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
):
    norm = normalize_answer(text, qtype)
    freq = freq_map.get(norm, 0) / max(1, pool_size)
    fmt = _format_score(text, norm, qtype)
    specificity_penalty = _specificity_penalty(text, qtype)
    unknown_penalty = UNKNOWN_PENALTY if norm == "Unknown" else 0.0
    anchor_bonus = ANCHOR_BONUS if anchor_norm and norm == anchor_norm else 0.0
    new_norm_penalty = NEW_NORM_PENALTY if init_norms and norm not in init_norms else 0.0
    return (
        fmt
        + (CONSENSUS_WEIGHT * freq)
        + anchor_bonus
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
):
    pool = candidates[:]
    rules, _ = rules_for(qtype)
    init_norms = {normalize_answer(t, qtype) for t in candidates}
    if not anchor_text and candidates:
        anchor_text = candidates[0]
    anchor_norm = normalize_answer(anchor_text, qtype) if anchor_text else None
    review_cache = review_cache or {}

    for step in range(TPO_STEPS):
        norm_pool = [normalize_answer(t, qtype) for t in pool]
        freq_map = Counter(norm_pool)
        review_scores = _review_scores_for_pool(
            reviewers, review_cache, question, qtype, pool, rules
        )
        scored = []
        for t, n in zip(pool, norm_pool):
            review = review_scores.get(n, 0.0)
            score = _score_candidate(
                t, qtype, freq_map, len(pool), anchor_norm, init_norms, review
            )
            scored.append((score, t, n, review))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = _select_top_k(scored, TOP_K, anchor_norm)

        print(f"Round {step + 1} pool (score | review_score | norm | text):")
        print(f"  Note: review_score = reviewer投票权重 (0.0=未选中, >0=被选中)")
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
            prompt = (
                f"{rules}\n"
                f"Question: {question}\n"
                f"Candidate answer: {text}\n"
                "Generate a concise alternative answer in the required format. "
                "Prefer a different plausible answer than the candidate when possible. "
                "Output only the answer."
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

        pool = [t for _, t, _, _ in selected] + new_candidates
        if anchor_text and anchor_norm and all(normalize_answer(t, qtype) != anchor_norm for t in pool):
            pool.append(anchor_text)
        if len(pool) > POOL_SIZE:
            norm_pool = [normalize_answer(t, qtype) for t in pool]
            freq_map = Counter(norm_pool)
            review_scores = _review_scores_for_pool(
                reviewers, review_cache, question, qtype, pool, rules
            )
            scored = []
            for t, n in zip(pool, norm_pool):
                review = review_scores.get(n, 0.0)
                score = _score_candidate(
                    t, qtype, freq_map, len(pool), anchor_norm, init_norms, review
                )
                scored.append((score, t, n, review))
            scored.sort(key=lambda x: x[0], reverse=True)
            pool = [t for _, t, _, _ in scored[:POOL_SIZE]]
            if anchor_text and anchor_norm and all(normalize_answer(t, qtype) != anchor_norm for t in pool):
                pool[-1] = anchor_text

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
policy = VLLMOpenAIMMPolicyV5("http://localhost:8000/v1", "google/medgemma-27b-it")
reviewer_policies = build_reviewers(REVIEWERS)
review_cache = {}

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
        refined_texts, refined_voted = tpo_optimize_text(
            policy,
            q,
            qtype,
            init_out.texts,
            anchor_text=init_voted,
            reviewers=reviewer_policies,
            review_cache=review_cache,
        )

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
        print(f"refined samples (after TPO, pool_size={len(refined_texts)}):")
        for i, t in enumerate(refined_texts):
            print(f"  {i}: {t}")
        print("refined voted:", refined_voted)
    print("final:", final_ans)

    seen += 1
    if seen >= MAX_QUESTIONS:
        break
