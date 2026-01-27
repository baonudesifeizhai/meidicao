from collections import Counter
import re
import json
from datetime import datetime

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
SPECIFICITY_PENALTY_LOBE = 0.05
SPECIFICITY_PENALTY_COMMA = 0.03
SPECIFICITY_PENALTY_MAX = 0.15

MAX_QUESTIONS = 50
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
        "name": "qwen2.5-vl-7b",
        "type": "vllm",
        "base_url": "http://localhost:8001/v1",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "weight": 1.0,
    },
    {
        "name": "biomistral",
        "type": "hf",
        "model": "BioMistral/BioMistral-7B",
        "weight": 1.0,
        "dtype": "bfloat16",
        "device_map": {"": "cuda:3"},
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


def _review_scores_for_pool(reviewers, cache, question, qtype, pool_texts, rules, image=None):
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
    if image is not None:
        prompt = (
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
    else:
        prompt = (
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
        prompt += f"{i}. {text}\n"
    prompt += "Return ONLY the number of the best answer."
    scores = {norm: 0.0 for norm in norms}
    weight_sum = 0.0
    for cfg, reviewer in reviewers:
        key = (cfg["name"], qtype, question, tuple(norms), image is not None)
        if key not in cache:
            try:
                raw = reviewer.generate(prompt, image=image)
            except NotImplementedError:
                if REVIEW_DEBUG:
                    print(f"  RM[{cfg['name']}] skipped (multimodal not supported)")
                continue
            choice = _parse_choice(raw, len(norms))
            if REVIEW_DEBUG:
                raw_display = " ".join((raw or "").split())
                choice_display = "None" if choice is None else str(choice)
                picked = "None" if choice is None else norms[choice - 1]
                img_tag = " [with image]" if image is not None else ""
                print(
                    f"  RM[{cfg['name']}] choice={choice_display} norm={picked} raw={raw_display!r}{img_tag}"
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
    high_consensus_anchor=False,  # Kept for compatibility but not used
):
    norm = normalize_answer(text, qtype)
    freq = freq_map.get(norm, 0) / max(1, pool_size)
    fmt = _format_score(text, norm, qtype)
    specificity_penalty = _specificity_penalty(text, qtype)
    unknown_penalty = UNKNOWN_PENALTY if norm == "Unknown" else 0.0
    # Never reward Unknown as anchor; it should be a fallback only.
    anchor_bonus = ANCHOR_BONUS if anchor_norm and norm == anchor_norm and norm != "Unknown" else 0.0
    # REMOVED: high consensus extra bonus - models can collectively be wrong
    # High consensus doesn't guarantee correctness
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
    image=None,
):
    pool = candidates[:]
    rules, _ = rules_for(qtype)
    init_norms = {normalize_answer(t, qtype) for t in candidates}
    if not anchor_text and candidates:
        anchor_text = candidates[0]
    anchor_norm = normalize_answer(anchor_text, qtype) if anchor_text else None
    # If anchor is Unknown but other answers exist, do not force-keep Unknown.
    if anchor_norm == "Unknown":
        if any(normalize_answer(t, qtype) != "Unknown" for t in candidates):
            anchor_norm = None
    review_cache = review_cache or {}
    
    # Check initial consensus: if anchor appears in 3+ out of 5 initial samples, it's likely correct
    # BUT: high consensus doesn't guarantee correctness (models can collectively be wrong)
    # So we use this only for early stopping, not for extra bonus
    init_norm_list = [normalize_answer(t, qtype) for t in candidates]
    anchor_freq_in_init = init_norm_list.count(anchor_norm) if anchor_norm else 0
    high_consensus_anchor = anchor_freq_in_init >= 3  # 3/5 = 60% consensus
    # REMOVED: high consensus bonus - models can collectively be wrong
    # Instead, we'll use early stopping if anchor is consistently leading

    for step in range(TPO_STEPS):
        norm_pool = [normalize_answer(t, qtype) for t in pool]
        freq_map = Counter(norm_pool)
        review_scores = _review_scores_for_pool(
            reviewers, review_cache, question, qtype, pool, rules, image=image
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
        if anchor_text and anchor_norm and all(normalize_answer(t, qtype) != anchor_norm for t in pool):
            pool.append(anchor_text)
        if len(pool) > POOL_SIZE:
            norm_pool = [normalize_answer(t, qtype) for t in pool]
            freq_map = Counter(norm_pool)
            review_scores = _review_scores_for_pool(
                reviewers, review_cache, question, qtype, pool, rules, image=image
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
            if anchor_text and anchor_norm and all(normalize_answer(t, qtype) != anchor_norm for t in pool):
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
    
    # Per question type
    by_type = {}
    for r in results:
        qtype = r.get("question_type", "unknown")
        if qtype not in by_type:
            by_type[qtype] = {"total": 0, "exact": 0, "partial": 0}
        by_type[qtype]["total"] += 1
        if exact_match(r["final_answer"], r["ground_truth"]):
            by_type[qtype]["exact"] += 1
        if partial_match(r["final_answer"], r["ground_truth"]):
            by_type[qtype]["partial"] += 1
    
    metrics = {
        "total": total,
        "exact_match": exact_matches,
        "exact_match_rate": exact_matches / total if total > 0 else 0.0,
        "partial_match": partial_matches,
        "partial_match_rate": partial_matches / total if total > 0 else 0.0,
        "by_type": {
            qtype: {
                "total": stats["total"],
                "exact_match_rate": stats["exact"] / stats["total"] if stats["total"] > 0 else 0.0,
                "partial_match_rate": stats["partial"] / stats["total"] if stats["total"] > 0 else 0.0,
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

policy = VLLMOpenAIMMPolicyV5("http://localhost:8000/v1", "google/medgemma-27b-it")
reviewer_policies = build_reviewers(REVIEWERS)

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
            anchor_text = anchor_out.texts[0] if anchor_out.texts else "Unknown"
            diverse_n = max(1, INIT_N - 1)
            diverse_out = policy.generate_n_mm(img, q, n=diverse_n, use_gate=True, temperature=1.2, top_p=0.85)
            init_texts = [anchor_text] + [t for t in diverse_out.texts if t]
            init_voted = anchor_text
            init_stats = _disagreement_stats(init_texts, qtype)
            if init_stats["unique"] < MIN_UNIQUE or init_stats["disagree"] < MIN_DISAGREE:
                continue

            refined_texts = init_texts
            refined_voted = init_voted
            if qtype not in YESNO_TYPES:
                # Use TPO for all questions to explore better answers
                # But be conservative: strongly protect the initial voted answer
                refined_texts, refined_voted = tpo_optimize_text(
                    policy,
                    q,
                    qtype,
                    init_texts,
                    anchor_text=init_voted,
                    reviewers=reviewer_policies,
                    review_cache=review_cache,
                    image=img,
                )

            final_ans = final_verify(policy, img, q, qtype, refined_voted)

            print("=" * 70)
            print(f"[{dataset_name}] Q{qi} type={qtype}")
            print("Q :", q)
            if gt is not None:
                print("GT:", gt)
            print("init samples:")
            for i, t in enumerate(init_texts):
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

            # Save results for validation
            result = {
                "dataset": dataset_name,
                "question_id": qi,
                "question": q,
                "question_type": qtype,
                "ground_truth": gt,
                "init_samples": init_texts,
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
        "results": all_results,
    }, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {output_file}")
print("You can use this file for validation and evaluation.")
