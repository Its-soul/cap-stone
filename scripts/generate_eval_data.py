import json
import random
import os
from datetime import datetime, timezone

def generate_track_a(num_worlds=16):
    pairs = []
    sources = ["D0", "D1", "D2"]
    
    for i in range(num_worlds):
        world_id = f"track_a-world{i}"
        
        # Pick 1 or 2 sources for the sum
        num_in_sum = random.choice([1, 2])
        in_sum = random.sample(sources, num_in_sum)
        out_sum = [s for s in sources if s not in in_sum]
        
        # Generate 4 pairs for this world
        # 2 positive (intervention on a source in sum)
        # 2 negative (intervention on a source out of sum)
        
        interventions = []
        labels = []
        if num_in_sum == 1:
            interventions.extend([in_sum[0], in_sum[0], out_sum[0], out_sum[1]])
            labels.extend([1, 1, 0, 0])
        else: # 2
            interventions.extend([in_sum[0], in_sum[1], out_sum[0], out_sum[0]])
            labels.extend([1, 1, 0, 0])
            
        # Shuffle them
        combined = list(zip(interventions, labels))
        random.shuffle(combined)
        
        base_vals = {s: random.randint(10, 50) for s in sources}
        
        for j, (interv_src, label) in enumerate(combined):
            pair_id = f"{world_id}-{interv_src}-Q0-{j}"
            val = base_vals[interv_src]
            
            premise = f"Snapshot {world_id}: input {interv_src} has numeric value {val}."
            hypothesis = f"In {world_id}, claim Q0 totals exactly {', '.join(sorted(in_sum))}; no other sources enter its sum."
            
            pair = {
                "derivation": {
                    "bindings": sorted(in_sum),
                    "intervention": {
                        "source": interv_src,
                        "before": val,
                        "after": val + 1,
                        "delta": 1
                    },
                    "rule": "source in declared transitive source-sum ancestry",
                    "topology": "cascade" if num_in_sum > 1 else "independent"
                },
                "hypothesis": hypothesis,
                "label": label,
                "pair_id": pair_id,
                "premise": premise,
                "source": "evaluation suite track a",
                "template_family": "track_a",
                "world_id": world_id
            }
            pairs.append(pair)
            
    return pairs

def generate_track_b(num_worlds=16):
    pairs = []
    sources = ["Alpha", "Beta", "Gamma"]
    
    for i in range(num_worlds):
        world_id = f"track_b-world{i}"
        
        num_in_sum = random.choice([1, 2])
        in_sum = random.sample(sources, num_in_sum)
        out_sum = [s for s in sources if s not in in_sum]
        
        interventions = []
        labels = []
        if num_in_sum == 1:
            interventions.extend([in_sum[0], in_sum[0], out_sum[0], out_sum[1]])
            labels.extend([1, 1, 0, 0])
        else: # 2
            interventions.extend([in_sum[0], in_sum[1], out_sum[0], out_sum[0]])
            labels.extend([1, 1, 0, 0])
            
        combined = list(zip(interventions, labels))
        random.shuffle(combined)
        
        base_vals = {s: random.randint(100, 500) for s in sources}
        
        for j, (interv_src, label) in enumerate(combined):
            pair_id = f"{world_id}-{interv_src}-Total-{j}"
            val = base_vals[interv_src]
            
            premise = f"Domain record {world_id}: Account {interv_src} shows a balance of ${val}."
            hypothesis = f"For the {world_id} audit, the final Total aggregates precisely {', '.join(sorted(in_sum))} without any external funds."
            
            pair = {
                "derivation": {
                    "bindings": sorted(in_sum),
                    "intervention": {
                        "source": interv_src,
                        "before": val,
                        "after": val + 10,
                        "delta": 10
                    },
                    "rule": "source in declared transitive source-sum ancestry",
                    "topology": "cascade" if num_in_sum > 1 else "independent"
                },
                "hypothesis": hypothesis,
                "label": label,
                "pair_id": pair_id,
                "premise": premise,
                "source": "evaluation suite track b",
                "template_family": "track_b",
                "world_id": world_id
            }
            pairs.append(pair)
            
    return pairs

def main():
    random.seed(42) # For reproducible eval data
    pairs = generate_track_a(16) + generate_track_b(16)
    
    # We create a new run directory for this evaluation phase
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_id = f"eval_{timestamp}_11111111"
    run_dir = os.path.join(r"D:\Code\Github\certificate-recovery\_private\runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    out_path = os.path.join(run_dir, "dependency_pairs.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p) + '\n')
            
    print(f"Generated {len(pairs)} pairs to {out_path}")
    
    # Let's also print the shortcut audit for this new dataset to prove it's balanced
    source_stats = {s: {'pos': 0, 'total': 0} for s in ["D0", "D1", "D2", "Alpha", "Beta", "Gamma"]}
    for p in pairs:
        s = p['derivation']['intervention']['source']
        source_stats[s]['total'] += 1
        if p['label'] == 1:
            source_stats[s]['pos'] += 1
            
    for s in ["D0", "D1", "D2", "Alpha", "Beta", "Gamma"]:
        stats = source_stats[s]
        if stats['total'] > 0:
            print(f"New Data Source {s}: {stats['pos']} pos / {stats['total']} total ({stats['pos']/stats['total']:.2%} pos)")
            
if __name__ == '__main__':
    main()
