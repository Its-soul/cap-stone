import json
from collections import defaultdict

def audit_shortcut(jsonl_path):
    print(f"Auditing {jsonl_path}...")
    source_stats = defaultdict(lambda: {'pos': 0, 'neg': 0, 'total': 0})
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            source = data['derivation']['intervention']['source']
            label = data['label']
            source_stats[source]['total'] += 1
            if label == 1:
                source_stats[source]['pos'] += 1
            else:
                source_stats[source]['neg'] += 1
                
    for source in sorted(source_stats.keys()):
        stats = source_stats[source]
        print(f"Source {source}: {stats['pos']} positive / {stats['total']} total ({stats['pos']/stats['total']:.2%} positive)")

if __name__ == '__main__':
    audit_shortcut(r'D:\Code\Github\certificate-recovery\_private\runs\pilot_20261005T095053_580194Z_02f39b02\dependency_pairs.jsonl')
