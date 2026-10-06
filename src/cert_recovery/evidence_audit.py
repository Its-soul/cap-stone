"""Read-only reconstruction of frozen evaluation evidence, without model loading."""
import json
from collections import defaultdict
from pathlib import Path

from .crypto import sha256
from .data_pipeline import read_jsonl, write_json, normalize_text
from .paired_metrics import rename_text, report_predictions, validate_pairs
from .pinned_smoke import sha256_file


def recover_robust_links(rows):
    """Post-hoc associations proved against IDs/text; never modify historical rows."""
    indexed = {r['pair_id']: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate historical IDs')
    aliases = ['Alpha', 'Beta', 'Gamma', 'Delta']
    pairs, excluded = [], []
    for world in sorted({r['world_id'] for r in rows}):
        for alias in aliases:
            a, b = [indexed[f'{world}-{alias}-{suffix}'] for suffix in ('pos', 'neg')]
            pairs.append({'comparison_id': f'{world}-{alias}-role', 'kind': 'role_change',
                          'members': [a['pair_id'], b['pair_id']], 'expected_labels': [1, 0],
                          'controlled': a['premise'] == b['premise'],
                          'changed_fields': [f for f in ('premise', 'hypothesis') if a[f] != b[f]],
                          'evidence': 'post-hoc exact authored world/alias/pos-neg IDs; context changes flagged'})
        for suffix in ('pos', 'neg'):
            a = indexed[f'{world}-Alpha-{suffix}']
            for alias in aliases[1:]:
                b = indexed[f'{world}-{alias}-{suffix}']
                mapping = dict(zip(aliases, [alias] + [x for x in aliases if x != alias]))
                pair = {'comparison_id': f'{world}-{alias}-{suffix}-rename', 'kind': 'renaming',
                        'members': [a['pair_id'], b['pair_id']], 'alias_map': mapping,
                        'expected_labels': [a['label'], b['label']], 'controlled': True,
                        'evidence': 'post-hoc exact alias-only text equivalence against generator ordering'}
                if all(normalize_text(rename_text(a[f], mapping)) == normalize_text(b[f]) for f in ('premise', 'hypothesis')):
                    pairs.append(pair)
                else:
                    excluded.append({**pair, 'reason': 'Non-alias text differs (metric/date); not a controlled renaming'})
    validate_pairs(rows, pairs)
    return {'schema_version': 1, 'association_status': 'INFERRED_POST_HOC_TEXT_VERIFIED',
            'dataset_canonical_sha256': sha256(rows), 'pairs': pairs, 'excluded_proposed_renamings': excluded}


def legacy_cohorts(rows, predictions):
    """Describe the historical calculation honestly, without calling cohorts pairs."""
    predictions = {p['pair_id']: p['prediction'] for p in predictions}
    worlds = defaultdict(list)
    for r in rows:
        worlds[r['world_id']].append(r)
    cohorts = [[r for r in members if r['label'] == label] for members in worlds.values() for label in (0, 1)]
    renaming = sum(len({predictions[r['pair_id']] for r in group}) == 1 for group in cohorts)
    role = sum(all(predictions[r['pair_id']] == r['label'] for r in group) for group in worlds.values())
    return {'within_label_world_homogeneity': {'numerator': renaming, 'denominator': len(cohorts),
             'unique_rows': len(rows), 'cohort_size': 4},
            'whole_world_perfect_classification': {'numerator': role, 'denominator': len(worlds),
             'unique_rows': len(rows), 'cohort_size': 8},
            'overlap_between_measures': len(rows), 'interpretation': 'legacy descriptive cohorts, not individual transformations'}


def fit_baselines(train):
    counts = defaultdict(lambda: [0, 0])
    for r in train:
        counts[r['derivation']['intervention']['source']][r['label']] += 1
    prior = sum(r['label'] for r in train) / len(train)
    return {'training_rows': len(train), 'training_hash': sha256(train), 'global_probability': prior,
            'identifier_probabilities': {k: v[1] / sum(v) for k, v in counts.items()},
            'identifier_label_counts': dict(counts), 'fit_scope': 'TRAIN ONLY'}


def baseline_predictions(rows, fitted, identifier=False):
    seen = fitted['identifier_probabilities']
    preds = []
    for r in rows:
        name = r['derivation']['intervention']['source']
        p = seen.get(name, fitted['global_probability']) if identifier else fitted['global_probability']
        preds.append({**r, 'probability': p, 'prediction': int(p >= .5)})
    coverage = sum(r['derivation']['intervention']['source'] in seen for r in rows)
    return preds, {'seen_identifier_rows': coverage, 'total_rows': len(rows), 'coverage': coverage / len(rows),
                   'unseen_fallback': 'training global positive prior, threshold 0.5',
                   'interpretation': 'constant unseen fallback does not establish absence of identifier shortcuts'}


def run_audit(root, out):
    root, out = Path(root), Path(out)
    if (out / 'audit.json').exists():
        raise FileExistsError('Audit already exists')
    pilot = root / '_private/runs/pilot_20261005T095053_580194Z_02f39b02'
    dataset = root / '_private/runs/eval_robust_20261005T111536_633570Z/dependency_pairs.jsonl'
    evaluation = dataset.parent.with_name(dataset.parent.name + '_evaluation')
    rows = read_jsonl(dataset)
    links = recover_robust_links(rows)
    write_json(out / 'paired_links.json', links)
    selection = json.loads((pilot / 'checkpoints/selection.json').read_text())
    actual_weight = sha256_file(pilot / 'checkpoints/best/model.safetensors')
    if actual_weight != selection['checkpoint_sha256']:
        raise ValueError('Original checkpoint hash mismatch')
    train_ids = {r['pair_id'] for r in read_jsonl(pilot / 'splits/train.jsonl')}
    train = [r for r in read_jsonl(pilot / 'dependency_pairs.jsonl') if r['pair_id'] in train_ids]
    fitted = fit_baselines(train)
    methods, legacy = {}, {}
    for method, filename in [('original', 'finetuned_evaluation'), ('nli_proxy', 'pretrained_baseline'),
                             ('constant', 'constant_baseline'), ('identifier', 'identifier_baseline')]:
        raw = json.loads((evaluation / 'metrics' / (filename + '.json')).read_text())
        saved = raw.get('test', raw)
        if 'dataset_hash' in raw and raw['dataset_hash'] != sha256(rows):
            raise ValueError('Historical canonical dataset hash differs')
        if method == 'original' and saved['threshold'] != selection['threshold']:
            raise ValueError('Original frozen threshold differs')
        if method == 'nli_proxy':
            pre = json.loads((pilot / 'metrics/pretrained_selection.json').read_text())
            if saved['threshold'] != pre['threshold']:
                raise ValueError('Original proxy frozen threshold differs')
        report = report_predictions(rows, saved['predictions'], links['pairs'], saved['threshold'])
        if report['confusion_matrix'] != saved['confusion_matrix'] or abs(report['macro_f1'] - saved['macro_f1']) > 1e-12:
            raise ValueError('Saved classification metrics differ from recomputation')
        if method in ('constant', 'identifier'):
            correct_preds, coverage = baseline_predictions(rows, fitted, method == 'identifier')
            if [p['prediction'] for p in correct_preds] != [p['prediction'] for p in saved['predictions']]:
                raise ValueError('Train-only baseline changes historical predictions')
            report['baseline_provenance'] = {**fitted, 'coverage': coverage,
                'historical_defect': 'old identifier fit used all 96 original rows; corrected fit uses 64 train rows; scores unchanged here'}
        methods[method] = report
        legacy[method] = legacy_cohorts(rows, saved['predictions'])
    tokenizer_files = {p.name: sha256_file(p) for p in (pilot / 'checkpoints/best').iterdir()
                       if p.is_file() and p.name != 'model.safetensors'}
    audit = {'schema_version': 1, 'warning': 'PRELIMINARY / EXPERIMENTAL', 'status': 'VERIFIED_OFFLINE',
             'dataset_file_sha256': sha256_file(dataset), 'dataset_canonical_sha256': sha256(rows),
             'unique_rows': len({(r['premise'], r['hypothesis']) for r in rows}),
             'checkpoint_sha256': actual_weight, 'original_threshold': selection['threshold'],
             'tokenizer_and_config_files_observed_now': tokenizer_files,
             'tokenizer_historical_attestation': 'No tokenizer file hashes stored by robust runner; current hashes are post-hoc evidence, not historical runtime attestation',
             'legacy_cohorts': legacy, 'methods': methods,
             'limitations': ['Role links identify authored counterparts; most alter context, not isolated interventions.',
                 'Historical nli_trap negative does not state an exhaustive dependency rule; annotations provisional.',
                 'Backup storage does not entail being fully operational or identical; historical NLI-entailment assertion unsupported.',
                 'Historical baseline elapsed_seconds=0 is unmeasured, not a measured zero.',
                 'No causal memorization claim is established by these scores.',
                 'Any deleted attempt evidence is unavailable; no reconstruction is presented as original execution evidence.']}
    write_json(out / 'audit.json', audit)
    return audit
