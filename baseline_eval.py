from collections import Counter
import re
import json
from datetime import datetime

from datasets import load_dataset

from tpo_mm_policy_v5 import (
    VLLMOpenAIMMPolicyV5,
    classify_question,
    normalize_answer,
)

YESNO_TYPES = {"contain_yesno", "healthy_yesno", "abnormal_yesno"}
TARGET_QTYPES = {"disease", "location", "short"}

MAX_QUESTIONS = 50
MAX_SCAN = 400

DATASETS = [
    ("SLAKE", "mdwiratathya/SLAKE-vqa-english", "test"),
    ("VQA-RAD", "flaviagiammarino/vqa-rad", "test"),
    ("PathVQA", "flaviagiammarino/path-vqa", "test"),
]


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
    
    exact_matches = sum(1 for r in results if exact_match(r["prediction"], r["ground_truth"]))
    partial_matches = sum(1 for r in results if partial_match(r["prediction"], r["ground_truth"]))
    
    # Per question type
    by_type = {}
    for r in results:
        qtype = r.get("question_type", "unknown")
        if qtype not in by_type:
            by_type[qtype] = {"total": 0, "exact": 0, "partial": 0}
        by_type[qtype]["total"] += 1
        if exact_match(r["prediction"], r["ground_truth"]):
            by_type[qtype]["exact"] += 1
        if partial_match(r["prediction"], r["ground_truth"]):
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


policy = VLLMOpenAIMMPolicyV5("http://localhost:8000/v1", "google/medgemma-27b-it")

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
                img = ex["image"]
                q = ex.get("question", ex.get("question_text", ""))
                gt = ex.get("answer", None)

            qtype = classify_question(q)
            if qtype in YESNO_TYPES or qtype not in TARGET_QTYPES:
                continue
            if qtype == "short" and not any(k in q.lower() for k in ["largest", "biggest", "main", "primary", "most", "dominant"]):
                continue

            # Baseline: single generation without TPO
            init_out = policy.generate_n_mm(img, q, n=1, use_gate=True, temperature=0.0, top_p=1.0)
            baseline_answer = init_out.voted

            # Save result
            result = {
                "dataset": dataset_name,
                "question_id": qi,
                "question": q,
                "question_type": qtype,
                "ground_truth": gt,
                "prediction": baseline_answer,
                "exact_match": exact_match(baseline_answer, gt) if gt else None,
                "partial_match": partial_match(baseline_answer, gt) if gt else None,
            }
            all_results[dataset_name].append(result)

            print(f"[{dataset_name}] Q{qi} type={qtype}")
            print(f"Q: {q}")
            if gt:
                print(f"GT: {gt}")
            print(f"Pred: {baseline_answer}")
            if gt:
                em = "✓" if exact_match(baseline_answer, gt) else "✗"
                pm = "✓" if partial_match(baseline_answer, gt) else "✗"
                print(f"Exact: {em} | Partial: {pm}")
            print("-" * 70)

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

# Calculate metrics
print("\n" + "=" * 70)
print("BASELINE EVALUATION RESULTS")
print("=" * 70)

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

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"baseline_results_{timestamp}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": timestamp,
        "model": "google/medgemma-27b-it",
        "method": "baseline_single_generation",
        "total_questions": total_seen,
        "dataset_stats": dataset_stats,
        "metrics": all_metrics,
        "overall_metrics": overall_metrics if all_results_flat else None,
        "results": all_results,
    }, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {output_file}")
