import json
import random
import os
from datetime import datetime, timezone
import hashlib

def normalize_text(text):
    return text.lower().replace(" ", "").replace(".", "").replace(",", "")

def check_overlap(pairs, old_pairs_path):
    if not os.path.exists(old_pairs_path):
        return 0, 0
    old_exact = set()
    old_norm = set()
    with open(old_pairs_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            old_exact.add((d['premise'], d['hypothesis']))
            old_norm.add((normalize_text(d['premise']), normalize_text(d['hypothesis'])))
            
    exact_overlap = 0
    norm_overlap = 0
    for p in pairs:
        ex = (p['premise'], p['hypothesis'])
        norm = (normalize_text(p['premise']), normalize_text(p['hypothesis']))
        if ex in old_exact: exact_overlap += 1
        if norm in old_norm: norm_overlap += 1
    return exact_overlap, norm_overlap

def generate_robust_suite():
    random.seed(12345)
    pairs = []
    
    # We will use exactly 4 identifiers to perfectly counterbalance across labels.
    identifiers = ["Alpha", "Beta", "Gamma", "Delta"]
    
    # 1. Arithmetic Sum paired renaming + role change
    for i in range(16):
        world = f"robust_arithmetic_{i}"
        # role change pair: Beta in sum (label=1) vs Beta out of sum (label=0)
        # We counterbalance by doing this for all identifiers.
        for target in identifiers:
            others = [x for x in identifiers if x != target]
            in_sum_1 = [target, others[0]]
            in_sum_0 = [others[1], others[2]]
            
            # Label 1 case (target is dependent)
            premise_1 = f"System state {world}: Component {target} reports a metric of {random.randint(10,99)}."
            hypothesis_1 = f"The final metric for Q_Total in {world} is computed exactly from {in_sum_1[0]} and {in_sum_1[1]}."
            pairs.append({
                "pair_id": f"{world}-{target}-pos", "world_id": world, "template_family": "arithmetic",
                "label": 1, "premise": premise_1, "hypothesis": hypothesis_1,
                "derivation": {"intervention": {"source": target}, "rule": "arithmetic sum"},
                "source": "robust_eval_suite", "status": "synthetic/provisional"
            })
            
            # Label 0 case (target is independent) - Role change
            premise_0 = f"System state {world}: Component {target} reports a metric of {random.randint(10,99)}."
            hypothesis_0 = f"The final metric for Q_Total in {world} is computed exactly from {in_sum_0[0]} and {in_sum_0[1]}."
            pairs.append({
                "pair_id": f"{world}-{target}-neg", "world_id": world, "template_family": "arithmetic",
                "label": 0, "premise": premise_0, "hypothesis": hypothesis_0,
                "derivation": {"intervention": {"source": target}, "rule": "arithmetic sum"},
                "source": "robust_eval_suite", "status": "synthetic/provisional"
            })

    # 2. Approval Chain (Varied wording and context order)
    for i in range(16):
        world = f"robust_approval_{i}"
        for target in identifiers:
            others = [x for x in identifiers if x != target]
            
            # Varied context order: Hypothesis first in logical structure (though we still pass premise/hypothesis to model)
            # Label 1
            premise_1 = f"Document {target} was signed on {random.randint(1,28)}/10/2026 in environment {world}."
            hypothesis_1 = f"Release requires sign-off from {target} and {others[0]}."
            pairs.append({
                "pair_id": f"{world}-{target}-pos", "world_id": world, "template_family": "approval",
                "label": 1, "premise": premise_1, "hypothesis": hypothesis_1,
                "derivation": {"intervention": {"source": target}, "rule": "approval chain"},
                "source": "robust_eval_suite", "status": "synthetic/provisional"
            })
            
            # Label 0
            premise_0 = f"Document {target} was signed on {random.randint(1,28)}/10/2026 in environment {world}."
            hypothesis_0 = f"Release requires sign-off from {others[1]} and {others[2]}."
            pairs.append({
                "pair_id": f"{world}-{target}-neg", "world_id": world, "template_family": "approval",
                "label": 0, "premise": premise_0, "hypothesis": hypothesis_0,
                "derivation": {"intervention": {"source": target}, "rule": "approval chain"},
                "source": "robust_eval_suite", "status": "synthetic/provisional"
            })

    # 3. NLI vs Dependency Trap
    # Hypothesis implies NLI entailment but NOT technical dependency.
    for i in range(16):
        world = f"robust_nli_trap_{i}"
        for target in identifiers:
            others = [x for x in identifiers if x != target]
            
            # True technical dependency (Label 1)
            premise_1 = f"Service {target} handles authentication for {world}."
            hypothesis_1 = f"The main API gateway depends on {target} to validate tokens."
            pairs.append({
                "pair_id": f"{world}-{target}-pos", "world_id": world, "template_family": "nli_trap",
                "label": 1, "premise": premise_1, "hypothesis": hypothesis_1,
                "derivation": {"intervention": {"source": target}, "rule": "gateway dependency"},
                "source": "robust_eval_suite", "status": "synthetic/provisional"
            })
            
            # NLI Entailment trap: Lexical overlap/entailment but NOT technical dependency (Label 0)
            # Example: The premise is that target is similar to main API, hypothesis says they are deployed together. 
            # This doesn't mean the main API depends on target's data output.
            premise_0 = f"Service {target} is a backup storage container in {world}."
            hypothesis_0 = f"Service {target} is fully operational and contains data identical to the main storage."
            pairs.append({
                "pair_id": f"{world}-{target}-neg", "world_id": world, "template_family": "nli_trap",
                "label": 0, "premise": premise_0, "hypothesis": hypothesis_0,
                "derivation": {"intervention": {"source": target}, "rule": "nli trap"},
                "source": "robust_eval_suite", "status": "synthetic/provisional"
            })

    return pairs

def main():
    pairs = generate_robust_suite()
    print(f"Generated {len(pairs)} pairs.")
    
    unique = set([(p['premise'], p['hypothesis']) for p in pairs])
    print(f"Unique model-visible pairs: {len(unique)}")
    
    # Verify counterbalancing
    stats = {label: {src: 0 for src in ["Alpha", "Beta", "Gamma", "Delta"]} for label in [0, 1]}
    for p in pairs:
        stats[p['label']][p['derivation']['intervention']['source']] += 1
    print("Identifier distribution by label (should be perfectly balanced):", stats)
    
    # Check overlap
    pilot_path = r"D:\Code\Github\certificate-recovery\_private\runs\pilot_20261005T095053_580194Z_02f39b02\dependency_pairs.jsonl"
    ex, norm = check_overlap(pairs, pilot_path)
    print(f"Overlap with old training: Exact={ex}, Normalized={norm}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_id = f"eval_robust_{timestamp}"
    run_dir = os.path.join(r"D:\Code\Github\certificate-recovery\_private\runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    out_path = os.path.join(run_dir, "dependency_pairs.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p) + '\n')
            
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
