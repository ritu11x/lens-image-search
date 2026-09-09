import json

with open("eval_set.json", "r") as f:
    eval_set = json.load(f)

precision_at_5_scores = []
precision_at_10_scores = []

print("=" * 60)
print("PER-QUERY RESULTS")
print("=" * 60)

for query, data in eval_set.items():
    candidates = data["candidates"]  # top-10, in ranked order
    relevant = set(data["relevant"])

    top_5 = candidates[:5]
    top_10 = candidates[:10]

    p_at_5 = len([c for c in top_5 if c in relevant]) / 5
    p_at_10 = len([c for c in top_10 if c in relevant]) / 10

    precision_at_5_scores.append(p_at_5)
    precision_at_10_scores.append(p_at_10)

    print(f"\nQuery: '{query}'")
    print(f"  Relevant marked: {len(relevant)}/10")
    print(f"  Precision@5:  {p_at_5:.2f}")
    print(f"  Precision@10: {p_at_10:.2f}")

avg_p5 = sum(precision_at_5_scores) / len(precision_at_5_scores)
avg_p10 = sum(precision_at_10_scores) / len(precision_at_10_scores)

print("\n" + "=" * 60)
print("OVERALL METRICS")
print("=" * 60)
print(f"Average Precision@5  across {len(eval_set)} queries: {avg_p5:.3f}")
print(f"Average Precision@10 across {len(eval_set)} queries: {avg_p10:.3f}")
