"""Future v3 execution behind external review/authorization gates; inert on import."""
import copy
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil

from .annotation_review import write_exclusive
from .contrast_data import transformation_metrics
from .crypto import sha256
from .data_pipeline import read_jsonl
from .model_pipeline import _prediction_report
from .pinned_smoke import sha256_file
from .v3_preparation import load_json, verify_proposal, explicit_roots, original_row, validate_links, V2


def utc_date(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('Explicit UTC review/authorization date required') from error
    if result.utcoffset() is None or result.utcoffset().total_seconds() != 0:
        raise ValueError('Review/authorization date must be UTC')


def validate_external_reviews(packet, approval, protocol):
    if (approval.get('record_kind') != 'EXTERNAL_HUMAN_REVIEW' or approval.get('human_reviewed') is not True
            or approval.get('status') != 'APPROVED' or approval.get('genuine_feedback_available') is not True):
        raise PermissionError('Genuine external human review remains pending; templates are not approvals')
    utc_date(approval.get('reviewed_at_utc'))
    if (approval.get('reviewed_protocol_sha256') != packet['contract_review']['reviewed_artifact_version']['protocol_sha256']
            or approval.get('reviewed_dataset_sha256') != protocol['dataset_file_sha256']):
        raise ValueError('Human review does not bind exact frozen protocol/datasets')
    mutable = {'reviewer_identity', 'decision', 'rationale', 'unresolved_ambiguity', 'human_review_status'}
    def check(expected, observed):
        if {k: v for k, v in expected.items() if k not in mutable} != {k: v for k, v in observed.items() if k not in mutable}:
            raise ValueError('Reviewed item/text/label/version differs; amend and refreeze, never rebind approval')
        if (observed.get('decision') != 'approve' or observed.get('human_review_status') != 'REVIEWED'
                or not isinstance(observed.get('reviewer_identity'), str) or not observed['reviewer_identity'].strip()
                or not isinstance(observed.get('rationale'), str) or not observed['rationale'].strip()
                or observed.get('unresolved_ambiguity') != ''):
            raise PermissionError('Every required item needs identified human approval, rationale and no unresolved ambiguity')
    check(packet['contract_review'], approval.get('contract_review', {}))
    for key in ['contrast_reviews', 'future_final_reviews']:
        expected = {r['item_id']: r for r in packet[key]}
        provided = approval.get(key, [])
        observed = {r['item_id']: r for r in provided}
        if len(observed) != len(provided) or set(observed) != set(expected):
            raise PermissionError('All 64 contrasts and 128 future-final rows require exact review coverage')
        for name, item in expected.items(): check(item, observed[name])
    return {'status': 'EXTERNAL_APPROVAL_VALIDATED', 'contrast_rows': 64, 'future_final_rows': 128,
            'provenance': 'Reviewer declarations supplied in external sidecar; not an automated human-review attestation.'}


def validate_authorization(spec_path, authorization):
    if (authorization.get('training_authorized') is not True or authorization.get('inference_authorized') is not True
            or authorization.get('future_final_inference_authorized') is not True):
        raise PermissionError('External authorization for training, validation inference and future-final comparison required')
    if not isinstance(authorization.get('authorized_by'), str) or not authorization['authorized_by'].strip():
        raise PermissionError('Identified training authorization required')
    utc_date(authorization.get('authorized_at_utc'))
    if authorization.get('launch_spec_sha256') != sha256_file(spec_path) or authorization.get('max_training_attempts') != 1:
        raise ValueError('Authorization must bind the exact launch specification and one training attempt')


def verify_prepared_bundle(root, bundle):
    root, bundle = Path(root), Path(bundle)
    frozen = load_json(bundle/'preparation_manifest.json')
    for name, digest in frozen['artifact_sha256'].items():
        if sha256_file(bundle/name) != digest: raise ValueError('Prepared artifact changed: '+name)
    for name, digest in frozen['source_sha256'].items():
        if sha256_file(root/name) != digest: raise ValueError('Launch/preparation source changed: '+name)
    spec = load_json(bundle/'launch_spec.json')
    original_protocol, originals, original_links = verify_proposal(root, spec['review_dir'])
    protocol = load_json(spec['protocol_path'])
    if spec['protocol_sha256'] != sha256_file(spec['protocol_path']):
        raise ValueError('Launch protocol binding differs')
    if protocol.get('parent_protocol_sha256') != spec['original_protocol_sha256']:
        raise ValueError('Replacement does not bind original proposal')
    rows = [r for split in ['train', 'validation', 'future_final'] for r in read_jsonl(bundle/'replacement'/f'{split}.jsonl')]
    links = load_json(bundle/'replacement/transformations.json')['links']
    if rows != [explicit_roots(r) for r in originals] or links != original_links:
        raise ValueError('Replacement changed more than the declared root-source wording')
    validate_links(rows, links)
    for split, digest in protocol['dataset_file_sha256'].items():
        if sha256_file(bundle/'replacement'/f'{split}.jsonl') != digest: raise ValueError('Replacement data changed')
    for name in ['controls', 'selection', 'acceptance_criteria', 'evaluation', 'group_assignments']:
        if protocol[name] != original_protocol[name]: raise ValueError('Replacement changed model/selection/guards')
    for key in ['controls', 'selection', 'acceptance_criteria', 'evaluation', 'dataset_file_sha256', 'group_assignments']:
        if spec[key] != protocol[key]: raise ValueError('Launch plan diverges from original frozen guards: '+key)
    if spec['execution_enabled'] is not False or spec['approved_dataset_protocol_versions'] is not None:
        raise ValueError('Prepared specification must remain disabled/unapproved; use external sidecars')
    audit = load_json(bundle/'tokenizer_audit.json')
    if (audit['status'] != 'PASS' or audit['truncated_rows'] or audit['decisive_information_failures']
            or audit['invalid_encoded_pairs'] or audit['batch_equivalence']['status'] != 'PASS'
            or audit['comparator_tokenizer_equivalence']['status'] != 'PASS' or spec['technical_blockers']):
        raise ValueError('Technical preflight remains blocked')
    for name, digest in audit['tokenizer_provenance']['files'].items():
        if sha256_file(Path(audit['tokenizer_provenance']['snapshot'])/name) != digest:
            raise ValueError('Pinned tokenizer bytes changed')
    if sha256_file(Path(spec['pinned_base']['path'])) != spec['pinned_base']['sha256']:
        raise ValueError('Pinned base weights changed')
    comparator = spec['frozen_v2_comparator']
    if sha256_file(Path(comparator['directory'])/'model.safetensors') != comparator['weights_sha256']:
        raise ValueError('Frozen comparator weights changed')
    if sha256_file(root/V2/'checkpoints/selection.json') != comparator['selection_sha256'] or comparator['threshold'] != .2:
        raise ValueError('Frozen comparator selection/threshold changed')
    for name, digest in comparator['tokenizer_hashes'].items():
        if sha256_file(Path(comparator['directory'])/name) != digest: raise ValueError('Comparator tokenizer changed')
    return spec, protocol, rows, links


def validate_launch_gates(root, bundle, approval_file, authorization_file):
    bundle = Path(bundle)
    # Human/authorization failure precedes package imports, weight checks and any output directory.
    spec = load_json(bundle/'launch_spec.json')
    packet = load_json(bundle/'mentor_packet.json')
    protocol = load_json(spec['protocol_path'])
    reviewed = validate_external_reviews(packet, load_json(approval_file), protocol)
    validate_authorization(bundle/'launch_spec.json', load_json(authorization_file))
    checked = verify_prepared_bundle(root, bundle)
    for package, required in spec['runtime_versions'].items():
        if importlib.metadata.version(package) != required:
            raise ValueError('Launch runtime differs from audited environment: '+package)
    return checked, reviewed


def initialize_attempt(root, bundle, attempt, approval_file, authorization_file):
    root, bundle, attempt = Path(root), Path(bundle), Path(attempt)
    (spec, protocol, rows, links), reviewed = validate_launch_gates(root, bundle, approval_file, authorization_file)
    if attempt.exists(): raise FileExistsError('Attempt directory exists; no overwrite or automatic retry')
    claim = {'attempt_directory': str(attempt.resolve()), 'launch_spec_sha256': sha256_file(bundle/'launch_spec.json'),
             'approval_sha256': sha256_file(approval_file), 'authorization_sha256': sha256_file(authorization_file),
             'utc': datetime.now(timezone.utc).isoformat()}
    write_exclusive(bundle/'launch.claim.json', claim)  # Atomic one-attempt guard.
    attempt.mkdir(parents=True, exist_ok=False)
    shutil.copytree(bundle/'prepared_splits', attempt/'splits')
    base = load_json(root/V2/'config.json')
    base['model'] = copy.deepcopy(spec['controls']['starting_model'])
    base['training'] = copy.deepcopy(spec['controls']['training'])
    base['data']['group_key'] = 'world_id'
    for key, name in [('processed', 'splits'), ('checkpoints', 'checkpoints'), ('results', 'metrics')]:
        base['paths'][key] = str((attempt/name).resolve())
    base['execution'] = {'allow_data_preparation': False, 'allow_model_inference': True,
                         'allow_training': True, 'allow_research_experiments': False}
    write_exclusive(attempt/'config.json', base)
    write_exclusive(attempt/'launch_binding.json', {**claim, 'config_sha256': sha256_file(attempt/'config.json'),
        'review_gate': reviewed, 'future_final_model_access': 'comparison only after selection_frozen.json',
        'source_bundle': str(bundle.resolve())})
    return spec


def future_metrics(rows, probabilities, threshold, captures, links):
    report = _prediction_report(rows, probabilities, threshold, 10, captures)
    validate_links(rows, links)
    base_rows = [original_row(r) for r in rows]
    projected = [{**r, 'prediction': p['prediction']} for r,p in zip(base_rows,report['predictions'])]
    report['transformations'] = transformation_metrics(base_rows, projected, links)
    report['false_alarm_rate'] = report['classification']['fp'] / sum(r['label'] == 0 for r in rows)
    report['missed_dependency_rate'] = report['classification']['fn'] / sum(r['label'] == 1 for r in rows)
    report['scenario_macro_f1'] = {}; report['category_recall'] = {}
    for scenario in sorted({r['scenario'] for r in rows}):
        selected = [i for i, r in enumerate(rows) if r['scenario'] == scenario]
        part = _prediction_report([rows[i] for i in selected], [probabilities[i] for i in selected], threshold, 10)
        report['scenario_macro_f1'][scenario] = part['macro_f1']
    for category in sorted({r['contrast_category'] for r in rows}):
        selected = [p for p in report['predictions'] if p['contrast_category'] == category]
        report['category_recall'][category] = {'numerator': sum(p['correct'] for p in selected),
                                               'denominator': len(selected), 'value': sum(p['correct'] for p in selected)/len(selected)}
    select = lambda p: p['contrast_category'] == 'entailment_dependency_0'
    independent = [p for p in report['predictions'] if select(p)]
    report['entailment_independent_false_alarm'] = {'numerator': sum(p['prediction'] == 1 for p in independent),
        'denominator': len(independent), 'value': sum(p['prediction'] == 1 for p in independent)/len(independent)}
    unchanged = [p for p in report['predictions'] if p['label'] == 1 and not p['derivation']['intervention']['value_changed']]
    report['unchanged_value_positive_recall'] = {'numerator': sum(p['prediction'] == 1 for p in unchanged),
        'denominator': len(unchanged), 'value': sum(p['prediction'] == 1 for p in unchanged)/len(unchanged)}
    report['worst_scenario_macro_f1'] = min(report['scenario_macro_f1'].values())
    report['per_evaluation_track'] = {}
    for track in sorted({r['evaluation_track'] for r in rows}):
        indexes = [i for i,r in enumerate(rows) if r['evaluation_track'] == track]
        part = _prediction_report([rows[i] for i in indexes], [probabilities[i] for i in indexes], threshold, 10)
        report['per_evaluation_track'][track] = {k: part[k] for k in ['classification','macro_f1','confusion_matrix','class_counts']}
    report['per_alias_position_value_change'] = {}
    projections = {'alias': lambda r: r['fields']['candidate'],
        'positive_support_position': lambda r: str(r['fields']['supports'].index(r['fields']['candidate'])) if r['label'] else 'excluded',
        'observed_value_change': lambda r: str(r['derivation']['intervention']['value_changed'])}
    for name, project in projections.items():
        report['per_alias_position_value_change'][name] = {}
        for value in sorted({project(r) for r in rows}):
            indexes = [i for i,r in enumerate(rows) if project(r) == value]
            part = _prediction_report([rows[i] for i in indexes], [probabilities[i] for i in indexes], threshold, 10)
            report['per_alias_position_value_change'][name][value] = {k: part[k] for k in ['classification','macro_f1','confusion_matrix','class_counts']}
    return report


def acceptance(candidate, comparator):
    guards = {
        'primary_false_alarm_reduction': comparator['entailment_independent_false_alarm']['value'] - candidate['entailment_independent_false_alarm']['value'] >= .20-1e-12,
        'overall_f1': candidate['macro_f1'] >= comparator['macro_f1']-.02-1e-12,
        'missed_dependencies': candidate['missed_dependency_rate'] <= comparator['missed_dependency_rate']+.05+1e-12,
        'worst_scenario': candidate['worst_scenario_macro_f1'] >= .70,
        'all_six_category_recall': min(c['value'] for c in candidate['category_recall'].values()) >= .70,
        'renaming_consistency': candidate['transformations']['renaming']['numerator']/candidate['transformations']['renaming']['denominator'] >= .95,
        'support_rule_both_correct': candidate['transformations']['support_rule']['numerator']/candidate['transformations']['support_rule']['denominator'] >= .80,
        'unchanged_value_dependency_recall': candidate['unchanged_value_positive_recall']['value'] >= .90}
    return {'guards': guards, 'all_pass': all(guards.values()),
            'interpretation': 'Predeclared joint criteria on provisional grouped synthetic data, not general accuracy, calibration or certificate safety.'}


def run_worker(root, bundle, attempt, stage, approval_file, authorization_file):
    root, bundle, attempt = Path(root), Path(bundle), Path(attempt)
    (spec, protocol, rows, links), _ = validate_launch_gates(root, bundle, approval_file, authorization_file)
    claim = load_json(bundle/'launch.claim.json'); binding = load_json(attempt/'launch_binding.json')
    if (claim['attempt_directory'] != str(attempt.resolve()) or binding['authorization_sha256'] != sha256_file(authorization_file)
            or binding['approval_sha256'] != sha256_file(approval_file) or binding['launch_spec_sha256'] != sha256_file(bundle/'launch_spec.json')
            or binding['config_sha256'] != sha256_file(attempt/'config.json')):
        raise ValueError('Worker does not bind the authorized attempt')
    if (attempt/stage/'finished.json').exists(): raise FileExistsError('Stage already completed or failed; no retry')
    config = load_json(attempt/'config.json')
    for name, digest in spec['prepared_splits']['file_sha256'].items():
        if sha256_file(attempt/'splits'/f'{name}.jsonl') != digest: raise ValueError('Attempt split changed')
    if stage not in {'training', 'comparison'}: raise ValueError('Unknown future execution stage')
    frozen = None
    if stage == 'comparison':
        training_state = load_json(attempt/'training/finished.json')
        if training_state['status'] != 'SUCCESS': raise ValueError('Successful training required')
        if sha256_file(attempt/'selection_frozen.json') != training_state['selection_frozen_sha256']:
            raise ValueError('Immutable validation selection freeze changed')
        frozen = load_json(attempt/'selection_frozen.json')
        if sha256_file(attempt/'checkpoints/selection.json') != frozen['selection_sha256']: raise ValueError('Validation selection changed')
        selection = load_json(attempt/'checkpoints/selection.json')
        if frozen['candidate_threshold'] != selection['threshold'] or frozen['comparator_threshold'] != .2:
            raise ValueError('Selected thresholds differ from immutable selection')
        for filename, digest in selection['checkpoint_file_hashes'].items():
            if sha256_file(attempt/'checkpoints/best'/filename) != digest: raise ValueError('Selected model/tokenizer bytes changed')
    import torch
    if torch.cuda.is_available(): raise ValueError('This launch is frozen for CPU; GPU requires a separately reviewed plan')
    torch.set_num_threads(4)
    if stage == 'training':
        from .model_pipeline import run_finetuning
        selection = run_finetuning(config, manual=True)
        if selection['actual_optimizer_steps'] != 256: raise ValueError('Actual training steps differ from frozen two-epoch budget')
        write_exclusive(attempt/'selection_frozen.json', {'selection_sha256': sha256_file(attempt/'checkpoints/selection.json'),
            'candidate_weights_sha256': selection['checkpoint_sha256'], 'candidate_threshold': selection['threshold'],
            'comparator_weights_sha256': spec['frozen_v2_comparator']['weights_sha256'], 'comparator_threshold': .2,
            'selection_policy': protocol['selection'], 'future_final_predictions_at_freeze': False})
        return {'status': 'SUCCESS', 'actual_optimizer_steps': 256, 'future_final_inference': False}
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from .model_pipeline import _infer
    final = [r for r in rows if r['split'] == 'future_final']; ids = {r['pair_id'] for r in final}
    final_links = [l for l in links if l['parent_pair_id'] in ids and l['child_pair_id'] in ids]
    results = {}
    for name, directory, digest, threshold in [
        ('candidate_v3', attempt/'checkpoints/best', frozen['candidate_weights_sha256'], frozen['candidate_threshold']),
        ('frozen_v2', Path(spec['frozen_v2_comparator']['directory']), frozen['comparator_weights_sha256'], .2)]:
        if sha256_file(directory/'model.safetensors') != digest: raise ValueError('Selected comparison weights changed')
        model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True, use_safetensors=True)
        if model.config.id2label != {0: 'independent', 1: 'depends_on'}: raise ValueError('Dependency labels differ')
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        captures = []; probabilities = _infer(config, final, model, tokenizer, 1, captures)
        results[name] = future_metrics(final, probabilities, threshold, captures, final_links)
        results[name]['execution_provenance'] = {'model_directory': str(directory), 'weights_sha256': digest,
            'threshold': threshold, 'source_protocol_sha256': spec['protocol_sha256'],
            'dataset_sha256': spec['dataset_file_sha256']['future_final'], 'input_fields': ['premise', 'hypothesis'],
            'runtime_versions': spec['runtime_versions'], 'device': 'CPU', 'source': 'future authorized direct binary model; no Space wrapper',
            'selection_frozen_sha256': sha256_file(attempt/'selection_frozen.json')}
        write_exclusive(attempt/'comparison'/f'{name}.json', results[name])
        del model, tokenizer
    write_exclusive(attempt/'comparison/acceptance.json', acceptance(results['candidate_v3'], results['frozen_v2']))
    return {'status': 'SUCCESS', 'future_final_rows_per_model': len(final), 'selection_retuned': False}
