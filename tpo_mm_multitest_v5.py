from datasets import load_dataset
from tpo_mm_policy_v5 import VLLMOpenAIMMPolicyV5

ds = load_dataset("mdwiratathya/SLAKE-vqa-english", split="test", streaming=True)
it = iter(ds)
policy = VLLMOpenAIMMPolicyV5("http://localhost:8000/v1", "medgemma-27b-it")

targets = {4,6,8}

for qi in range(10):
    ex = next(it)
    if qi not in targets:
        continue
    img = ex["image"]
    q = ex["question"]
    gt = ex.get("answer", None)

    out = policy.generate_n_mm(img, q, n=5)

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
