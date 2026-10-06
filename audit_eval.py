import json
from collections import defaultdict
p = r'_private/runs/eval_20261005T105005_332601Z_11111111/dependency_pairs.jsonl'
stats = defaultdict(lambda: {'pos': 0, 'neg': 0, 'total': 0})
unique = set()
for line in open(p):
    d = json.loads(line)
    src = d['derivation']['intervention']['source']
    label = d['label']
    stats[src]['total'] += 1
    if label == 1: stats[src]['pos'] += 1
    else: stats[src]['neg'] += 1
    unique.add((d['premise'], d['hypothesis']))
print(f'Unique pairs: {len(unique)}')
for src in sorted(stats.keys()):
    s = stats[src]
    print(f'Source {src}: {s["pos"]} positive / {s["neg"]} negative / {s["total"]} total')
