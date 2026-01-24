from datasets import load_dataset
from tpo_mm_policy_v5 import VLLMOpenAIMMPolicyV5, classify_question

ds = load_dataset("mdwiratathya/SLAKE-vqa-english", split="test", streaming=True)
it = iter(ds)
policy = VLLMOpenAIMMPolicyV5("http://localhost:8000/v1", "medgemma-27b-it")

targets = None
max_questions = 20
max_non_yesno = 12
seen = 0
seen_non_yesno = 0

for qi in range(10):
    ex = next(it)
    if targets is not None and qi not in targets:
        continue
    img = ex["image"]
    q = ex["question"]
    gt = ex.get("answer", None)

    qtype = classify_question(q)
    if qtype in ("contain_yesno", "healthy_yesno"):
        if seen_non_yesno < max_non_yesno:
            continue
    else:
        seen_non_yesno += 1

    seen += 1
    if seen > max_questions:
        break

    out = policy.generate_n_mm(img, q, n=5, use_gate=True, temperature=1.2, top_p=0.85)

    print("=" * 70)
    print(f"[Q{qi}] type={out.qtype}")
    print("Q :", q)
    if gt is not None:
        print("GT:", gt)
    print("samples:")
    for i, t in enumerate(out.texts):
        print(f"  {i}: {t}")
    print("VOTED:", out.voted)
    print("RAW:", out.raw)
