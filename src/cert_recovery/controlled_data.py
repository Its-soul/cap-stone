"""Provisional synthetic dependency examples with explicit rules and transformations."""
from collections import Counter, defaultdict
import random
import re

from .crypto import sha256
from .data_pipeline import normalize_text
from .engine import CertificateSystem
from .models import ArtifactKind as K, VerificationResult
from .paired_metrics import rename_text, validate_pairs

SEED = 20261005
ALIASES = {'train': ['Oak', 'Pine', 'Elm', 'Fir'],
           'validation': ['Oak', 'Pine', 'Lime', 'Reed'],
           'test': ['Elm', 'Fir', 'Ash', 'Yew']}
TEMPLATES = [
    ('Snapshot {world}: source {source} records {value}.', 'Task {world} uses this complete rule: {rule}.'),
    ('In ledger {world}, input {source} has value {value}.', 'The entire specification for result {world} is {rule}.'),
    ('For batch {world}, the reading from {source} equals {value}.', 'Batch output {world} is defined exclusively by {rule}.'),
    ('Register {world} assigns {value} to entry {source}.', 'Register outcome {world} follows only this instruction: {rule}.'),
    ('At site {world}, sensor {source} reports {value}.', 'Site calculation {world} follows exactly {rule}.'),
]


def rule_value(scenario, values, required):
    if scenario in ('approval_all', 'authentication_all'):
        return int(all(values[name] for name in required))
    weights = [2, 3] if scenario == 'weighted_sum' else [1, 1]
    return sum(values[name] * weight for name, weight in zip(required, weights))


def witness(scenario, values, required, candidate):
    """Witness a declared function through the unchanged deterministic core."""
    system = CertificateSystem()
    refs = {name: system.add_dependency(name, {'value': value}) for name, value in values.items()}
    def evaluate(nodes):
        roots = {n.ref.artifact_id: n.payload['value'] for n in nodes if n.kind == K.DEPENDENCY}
        return rule_value(scenario, roots, required)
    system.register_verifier('declared-rule', lambda subject, ancestry:
        VerificationResult(subject.payload['value'] == evaluate(ancestry), 'exact declared function'))
    claim = system.add_artifact('Q', K.CLAIM, {'value': rule_value(scenario, values, required)},
                                tuple(refs[name] for name in required))
    cert = system.issue_certificate('C', claim, 'declared-rule')
    if not system.is_valid(cert):
        raise AssertionError('Initial deterministic support invalid')
    system.register_rebuilder('Q', lambda old, supports: {'value': evaluate(supports)})
    before = system.latest('Q').payload['value']
    delta = 0 if scenario in ('approval_all', 'authentication_all') else values[candidate] + 1
    change = system.change_dependency(candidate, {'value': delta})
    system.execute_recovery(system.plan_recovery(change))
    after = system.latest('Q').payload['value']
    return {'source': candidate, 'before': before, 'after': after,
            'changed_value': delta, 'claim_version_after': system.latest('Q').ref.version,
            'certificate_valid_after': system.is_valid(system.latest('C').ref)}


def _rule(scenario, required):
    a, b = required
    if scenario == 'approval_all':
        return f'sign-off is granted if and only if documents {a} and {b} are both signed'
    if scenario == 'authentication_all':
        return f'access is allowed if and only if services {a} and {b} both validate tokens'
    if scenario == 'weighted_sum':
        return f'the numeric sum of twice {a} and three times {b}, with all other sources excluded'
    return f'the numeric sum of {a} and {b}, with all other sources excluded'


def generate_controlled():
    rng = random.Random(SEED)
    rows, pairs, assignments = [], [], {}
    scenarios = ['sum', 'approval_all', 'authentication_all', 'nli_context']
    for split, aliases in ALIASES.items():
        families = [(s, t) for s in scenarios for t in ([0, 1] if split == 'train' else [2 if split == 'validation' else 3])]
        if split == 'test':
            families += [('weighted_sum', 3), ('weighted_sum', 4)]
        for scenario, template in families:
            family = f'{scenario}-t{template}'
            assignments[family] = split
            for block in range(2 if split == 'train' else 1):
                for index, candidate in enumerate(aliases):
                    world = f'w{len(pairs):04d}'
                    others = [a for a in aliases if a != candidate]
                    facts = {a: (1 if scenario in ('approval_all', 'authentication_all') else rng.randint(2, 8)) for a in aliases}
                    # Each source occurs in both labels and both required-list positions.
                    required_sets = [[candidate, others[0]], [others[1], others[2]]]
                    if (index + block) % 2:
                        required_sets = [list(reversed(x)) for x in required_sets]
                    quartet = []
                    mapping = {aliases[j]: aliases[(j+1) % 4] for j in range(4)}
                    for renamed in (False, True):
                        for label in (1, 0):
                            required = required_sets[1-label]
                            value = ('signed' if scenario == 'approval_all' else 'valid' if scenario == 'authentication_all' else str(facts[candidate]))
                            rule = _rule(scenario, required)
                            premise = TEMPLATES[template][0].format(world=world, source=candidate, value=value)
                            hypothesis = TEMPLATES[template][1].format(world=world, rule=rule)
                            nli = 'undetermined: rule not asserted in premise'
                            if scenario == 'nli_context':
                                premise += ' Complete task rule: ' + rule + '.'
                                hypothesis += ' ' + TEMPLATES[template][0].format(world=world, source=candidate, value=value)
                                nli = 'entailment by repeated literal rule and fact, for both dependency labels'
                            derivation = witness(scenario, facts, required, candidate)
                            if bool(derivation['before'] != derivation['after']) != bool(label) or not derivation['certificate_valid_after']:
                                raise AssertionError('Provisional annotation disagrees with deterministic function')
                            if renamed:
                                premise, hypothesis = rename_text(premise, mapping), rename_text(hypothesis, mapping)
                                required = [mapping[x] for x in required]
                                facts_for_row = {mapping[k]: v for k, v in facts.items()}
                                derivation = {**derivation, 'source': mapping[candidate]}
                            else:
                                facts_for_row = facts
                            row = {'pair_id': f'{family}-{world}-r{int(renamed)}-y{label}', 'world_id': world,
                                   'template_family': family, 'scenario': scenario, 'split': split,
                                   'premise': premise, 'hypothesis': hypothesis, 'label': label,
                                   'source': 'synthetic controlled v2; provisional; not human-reviewed',
                                   'derivation': {'rule': rule if not renamed else rename_text(rule, mapping),
                                                  'required_sources': required, 'world_values': facts_for_row,
                                                  'intervention': derivation, 'expected_nli_reasoning': nli,
                                                  'candidate_required_position': required.index(derivation['source']) if label else None}}
                            rows.append(row)
                            quartet.append(row)
                    for a, b, kind in [(0, 2, 'renaming'), (1, 3, 'renaming'), (0, 1, 'role_change'), (2, 3, 'role_change')]:
                        x, y = quartet[a], quartet[b]
                        pairs.append({'comparison_id': x['pair_id'] + '-to-' + y['pair_id'], 'kind': kind,
                                      'members': [x['pair_id'], y['pair_id']], 'expected_labels': [x['label'], y['label']],
                                      'alias_map': mapping if kind == 'renaming' else None, 'controlled': True,
                                      'context_rule_in_premise': scenario == 'nli_context',
                                      'changed_fields': [f for f in ('premise', 'hypothesis') if x[f] != y[f]],
                                      'evidence': 'authored controlled transformation; declared rule and witness'})
    validate_pairs(rows, pairs)
    return rows, pairs, assignments


def leakage_audit(rows, pairs, stress_rows=()):
    validate_pairs(rows, pairs)
    by_id = {r['pair_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate IDs')
    for p in pairs:
        if len({by_id[k]['split'] for k in p['members']}) != 1:
            raise ValueError('Related transformation crosses splits')
    aliases = sorted({a for names in ALIASES.values() for a in names}, key=len, reverse=True)
    def normalized(r):
        return tuple(normalize_text(r[f]).casefold() for f in ('premise', 'hypothesis'))
    def structure(r):
        def skeleton(s):
            s = re.sub(r'\b(?:' + '|'.join(aliases) + r')\b', '<alias>', s)
            s = re.sub(r'\bw\d+\b', '<world>', s)
            s = re.sub(r'\b\d+\b', '<number>', s)
            return normalize_text(s).casefold()
        return tuple(skeleton(r[f]) for f in ('premise', 'hypothesis'))
    splits = {name: [r for r in rows if r['split'] == name] for name in ALIASES}
    for name, subset in splits.items():
        if len({normalized(r) for r in subset}) != len(subset):
            raise ValueError('Duplicate normalized input pairs')
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        for projection in (normalized, structure):
            if {projection(r) for r in splits[a]} & {projection(r) for r in splits[b]}:
                raise ValueError('Normalized or structural wording-template leakage')
        if {r['template_family'] for r in splits[a]} & {r['template_family'] for r in splits[b]}:
            raise ValueError('Template family leakage')
    if {normalized(r) for r in rows} & {normalized(r) for r in stress_rows}:
        raise ValueError('Stress benchmark text reused')
    worlds = defaultdict(set)
    for row in rows:
        worlds[row['world_id']].add(row['split'])
    if any(len(s) != 1 for s in worlds.values()):
        raise ValueError('World variants cross split')
    return {'status': 'PASS', 'seed': SEED, 'normalized_duplicates': 0, 'cross_split_structural_template_overlaps': 0,
            'cross_split_world_overlaps': 0, 'stress_text_overlap': 0,
            'split_rows': {k: len(v) for k, v in splits.items()},
            'split_hashes': {k: sha256(v) for k, v in splits.items()},
            'template_groups': {k: sorted({r['template_family'] for r in v}) for k, v in splits.items()},
            'scenarios': {k: dict(Counter(r['scenario'] for r in v)) for k, v in splits.items()},
            'scenario_overlap': 'Four rules intentionally shared; wording templates disjoint; weighted_sum only final test',
            'identifier_counts': {k: dict(Counter((r['derivation']['intervention']['source'] + ':' + str(r['label'])) for r in v)) for k, v in splits.items()},
            'positive_positions': {k: dict(Counter(str(r['derivation']['candidate_required_position']) for r in v if r['label'])) for k, v in splits.items()},
            'limitations': 'Closed synthetic function sensitivity is provisional task dependency, not real-world causal necessity. Structural check removes aliases/world IDs/numbers; shared scenario semantics are intentional.'}
