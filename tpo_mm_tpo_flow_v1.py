from datasets import load_dataset

from tpo_mm_policy_v5 import (
    VLLMOpenAIMMPolicyV5,
    classify_question,
    normalize_answer,
    rules_for,
)

YESNO_TYPES = {"contain_yesno", "healthy_yesno", "abnormal_yesno"}

INIT_N = 5
REFINE_N = 5
REFINE_STEPS = 2
REFINE_TEMPERATURE = 0.9
REFINE_TOP_P = 0.9
FINAL_TEMPERATURE = 0.0

MAX_QUESTIONS = 8
MAX_NON_YESNO = 6
MAX_SCAN = 200


def vote_texts(texts, qtype):
    norm = [normalize_answer(t, qtype) for t in texts]
    voted = max(norm, key=norm.count)
    return voted, norm


def refine_text_only(policy, question, qtype, candidates):
    cur = candidates[:]
    rules, _ = rules_for(qtype)
    for _ in range(REFINE_STEPS):
        prompt = (
            f"{rules}\n"
            f"Question: {question}\n"
            "Candidate answers:\n"
            + "\n".join(f"- {c}" for c in cur)
            + "\nPick the best answer or improve it. Output only the answer."
        )
        cur = policy.generate_n_text(
            prompt,
            n=REFINE_N,
            temperature=REFINE_TEMPERATURE,
            top_p=REFINE_TOP_P,
            max_tokens=32,
        )
    return cur


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
seen_non_yesno = 0

for qi in range(MAX_SCAN):
    ex = next(it)
    img = ex["image"]
    q = ex["question"]
    gt = ex.get("answer", None)

    qtype = classify_question(q)
    if qtype in YESNO_TYPES:
        if seen_non_yesno < MAX_NON_YESNO:
            continue
    else:
        seen_non_yesno += 1

    init_out = policy.generate_n_mm(img, q, n=INIT_N, use_gate=True, temperature=1.2, top_p=0.85)
    init_voted = init_out.voted

    refined_texts = init_out.texts
    refined_voted = init_voted
    if qtype not in YESNO_TYPES:
        refined_texts = refine_text_only(policy, q, qtype, init_out.texts)
        refined_voted, _ = vote_texts(refined_texts, qtype)

    final_ans = final_verify(policy, img, q, qtype, refined_voted)

    print("=" * 70)
    print(f"[Q{qi}] type={qtype}")
    print("Q :", q)
    if gt is not None:
        print("GT:", gt)
    print("init samples:")
    for i, t in enumerate(init_out.texts):
        print(f"  {i}: {t}")
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
