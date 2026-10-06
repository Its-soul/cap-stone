"""One explicitly executed, bounded candidate experiment; no work on import."""
from datetime import datetime, timezone
import gc
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
from time import perf_counter
import uuid

from .controlled_data import generate_controlled, leakage_audit
from .crypto import canonical_json, sha256
from .data_pipeline import prepare_data, read_jsonl, write_json, load_verified_splits
from .evidence_audit import fit_baselines, baseline_predictions
from .model_pipeline import _infer, _prediction_report, run_finetuning, run_pretrained_baseline
from .paired_metrics import report_predictions
from .pinned_smoke import sha256_file, validate_label_order

PILOT = '_private/runs/pilot_20261005T095053_580194Z_02f39b02'
STRESS = '_private/runs/eval_robust_20261005T111536_633570Z/dependency_pairs.jsonl'


def freeze(root, audit_dir, offline_checks):
    root, audit_dir = Path(root), Path(audit_dir)
    audit = json.loads((audit_dir / 'audit.json').read_text())
    if audit['status'] != 'VERIFIED_OFFLINE' or offline_checks['status'] != 'PASS':
        raise ValueError('Corrected audit and all offline checks required before protocol freeze')
    out = root / '_private/runs' / ('candidate_v2_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_') + uuid.uuid4().hex[:8])
    out.mkdir(exist_ok=False)
    rows, pairs, assignments = generate_controlled()
    leak = leakage_audit(rows, pairs, read_jsonl(root / STRESS))
    (out / 'dependency_pairs.jsonl').write_text(''.join(canonical_json(r) + '\n' for r in rows), encoding='utf-8')
    config = json.loads((root / PILOT / 'config.json').read_text())
    for key, name in [('raw_pairs', 'dependency_pairs.jsonl'), ('processed', 'splits'), ('checkpoints', 'checkpoints'), ('results', 'metrics'), ('workflow_cases', 'unused.jsonl')]:
        config['paths'][key] = str(out / name)
    config['execution']['allow_research_experiments'] = False
    write_json(out / 'config.json', config)
    split_manifest = prepare_data(config, manual=True, group_assignments=assignments)
    write_json(out / 'paired_links.json', {'schema_version': 1, 'pairs': pairs, 'association_status': 'AUTHORED_TRANSFORMATIONS'})
    write_json(out / 'leakage_audit.json', leak)
    write_json(out / 'offline_preflight.json', offline_checks)
    protocol = {'schema_version': 1, 'warning': 'PRELIMINARY / EXPERIMENTAL', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'FROZEN_BEFORE_TRAINING', 'audit_dir': str(audit_dir),
                'model': config['model'], 'training': config['training'], 'training_source': 'fresh pinned base, new seeded two-class head; no original trained weights',
                'dataset_file_sha256': sha256_file(out / 'dependency_pairs.jsonl'),
                'dataset_canonical_sha256': sha256(rows), 'split_manifest_sha256': sha256(split_manifest),
                'config_sha256': sha256(config),
                'split_file_sha256': {name: sha256_file(out / 'splits' / (name + '.jsonl')) for name in ('train', 'validation', 'test')},
                'group_assignments': assignments, 'leakage_checks': leak,
                'stress_status': 'UNCHANGED DEVELOPMENT STRESS BENCHMARK; known failures informed design',
                'stress_file_sha256': sha256_file(root / STRESS), 'original_checkpoint_sha256': audit['checkpoint_sha256'],
                'selection': {'checkpoint': 'validation macro F1 at each epoch, earliest best retained',
                              'threshold': 'validation positive F1, recall, then smallest candidate; no stress/final selection'},
                'budget': {'training_attempts': 1, 'training_timeout_seconds': 1800, 'stage_timeout_seconds': 1800,
                           'epochs': 2, 'max_optimizer_steps': 256, 'torch_threads': 4, 'device': 'local CPU', 'paid_compute': False},
                'annotations': 'Synthetic closed-function sensitivity, provisional, no human review',
                'authority': 'Semantic outputs advisory; no graph edge creation or deterministic verifier changes'}
    source = out / 'source_snapshot'
    source.mkdir()
    for path in [*sorted((root / 'src/cert_recovery').glob('*.py')), root / 'scripts/run_candidate_v2.py', root / 'requirements-colab.txt']:
        shutil.copy2(path, source / path.name)
    protocol['source_files_sha256'] = {p.name: sha256_file(p) for p in source.iterdir()}
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'manifest.json', {'status': 'PREPARED', 'run_id': out.name, 'protocol_sha256': sha256(protocol), 'stages': {}})
    return out


def verify_frozen(run):
    run = Path(run)
    protocol = json.loads((run / 'protocol.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['protocol_sha256'] != sha256(protocol):
        raise ValueError('Frozen protocol changed')
    config = json.loads((run / 'config.json').read_text())
    if sha256(config) != protocol['config_sha256']:
        raise ValueError('Frozen configuration changed')
    splits, sm = load_verified_splits(config)
    if sha256(sm) != protocol['split_manifest_sha256'] or sha256_file(run / 'dependency_pairs.jsonl') != protocol['dataset_file_sha256']:
        raise ValueError('Frozen data changed')
    for name, digest in protocol['split_file_sha256'].items():
        if sha256_file(run / 'splits' / (name + '.jsonl')) != digest:
            raise ValueError('Frozen split bytes changed')
    for name, digest in protocol['source_files_sha256'].items():
        if sha256_file(run / 'source_snapshot' / name) != digest:
            raise ValueError('Source snapshot changed')
        actual = Path(config['project_root']) / ('src/cert_recovery/' + name if name.endswith('.py') and name != 'run_candidate_v2.py' else 'scripts/' + name if name == 'run_candidate_v2.py' else name)
        if sha256_file(actual) != digest:
            raise ValueError('Current implementation differs from frozen source')
    return config, protocol


def execute_stage(run, stage):
    run = Path(run)
    config, protocol = verify_frozen(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    if stage in manifest['stages']:
        raise FileExistsError('Stage already attempted; preserve attempt and choose a separately authorized new run')
    if stage != 'training' and manifest['stages'].get('training', {}).get('status') != 'SUCCESS':
        raise ValueError('Successful selected training required')
    if stage == 'comparison' and manifest['stages'].get('proxy_selection', {}).get('status') != 'SUCCESS':
        raise ValueError('Freeze proxy selection before comparisons')
    attempt = run / ('attempt_' + stage + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_') + uuid.uuid4().hex[:8])
    attempt.mkdir(exist_ok=False)
    start = perf_counter()
    state = {'status': 'RUNNING', 'attempt': str(attempt), 'started_utc': datetime.now(timezone.utc).isoformat()}
    manifest['stages'][stage] = state
    write_json(run / 'manifest.json', manifest)
    write_json(attempt / 'status.json', state)
    try:
        import torch
        torch.set_num_threads(protocol['budget']['torch_threads'])
        if torch.cuda.is_available():
            raise ValueError('This frozen experiment is local CPU only')
        if stage == 'training':
            run_finetuning(config, manual=True)
        elif stage == 'proxy_selection':
            run_pretrained_baseline(config, manual=True, include_test=False)
        elif stage == 'comparison':
            compare(run, config, protocol)
        else:
            raise ValueError('Unknown stage')
        state['status'] = 'SUCCESS'
    except BaseException as error:
        state['status'] = 'ERROR'
        state['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        state['elapsed_seconds'] = perf_counter() - start
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(attempt / 'status.json', state)
        manifest['stages'][stage] = state
        manifest['status'] = 'SUCCESS' if stage == 'comparison' and state['status'] == 'SUCCESS' else state['status']
        write_json(run / 'manifest.json', manifest)
    return state


def compare(run, config, protocol):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    root = Path(config['project_root'])
    selection = json.loads((run / 'checkpoints/selection.json').read_text())
    for name, digest in selection['checkpoint_file_hashes'].items():
        if sha256_file(run / 'checkpoints/best' / name) != digest:
            raise ValueError('Selected model/tokenizer bytes changed')
    original_selection = json.loads((root / PILOT / 'checkpoints/selection.json').read_text())
    proxy_selection = json.loads((run / 'metrics/pretrained_selection.json').read_text())
    write_json(run / 'comparison_selection_frozen.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_selection_sha256': sha256(selection), 'original_selection_sha256': sha256(original_selection),
        'proxy_selection_sha256': sha256(proxy_selection), 'final_predictions_not_yet_inspected': True})
    raw = read_jsonl(run / 'dependency_pairs.jsonl')
    all_links = json.loads((run / 'paired_links.json').read_text())['pairs']
    stress = read_jsonl(root / STRESS)
    stress_links = json.loads((Path(protocol['audit_dir']) / 'paired_links.json').read_text())['pairs']
    datasets = {'validation': [r for r in raw if r['split'] == 'validation'], 'stress': stress,
                'final': [r for r in raw if r['split'] == 'test']}
    links = {name: stress_links if name == 'stress' else [p for p in all_links if set(p['members']) <= {r['pair_id'] for r in rows}]
             for name, rows in datasets.items()}
    fitted = fit_baselines([r for r in raw if r['split'] == 'train'])
    write_json(run / 'baseline_fit.json', fitted)
    results = {name: {} for name in datasets}
    for method, source, threshold in [('original', root / PILOT / 'checkpoints/best', original_selection['threshold']),
                                      ('candidate_v2', run / 'checkpoints/best', selection['threshold']),
                                      ('nli_proxy', config['model']['pretrained_id'], proxy_selection['threshold'])]:
        if method == 'original' and sha256_file(source / 'model.safetensors') != protocol['original_checkpoint_sha256']:
            raise ValueError('Frozen original comparator changed')
        kwargs = {'local_files_only': True, 'trust_remote_code': False}
        if method == 'nli_proxy':
            kwargs['revision'] = config['model']['revision']
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(source, use_safetensors=True, **kwargs)
        positive = 1
        if method == 'nli_proxy':
            validate_label_order(model.config.to_dict(), ['contradiction', 'entailment', 'neutral'])
            if model.config._commit_hash != config['model']['revision']:
                raise ValueError('Proxy immutable revision mismatch')
        for name, rows in datasets.items():
            lengths = [len(tokenizer(r['premise'], r['hypothesis'], truncation=False)['input_ids']) for r in rows]
            if any(n > config['model']['max_length'] for n in lengths):
                raise ValueError('Comparison task context truncated')
            capture = []
            start = perf_counter()
            probs = _infer(config, rows, model, tokenizer, positive, capture)
            elapsed = perf_counter() - start
            base = _prediction_report(rows, probs, threshold, 10, capture)
            report = report_predictions(rows, base['predictions'], links[name], threshold)
            report['predictions'] = base['predictions']
            report['provenance'] = {'source': str(source), 'requested_revision': config['model']['revision'] if method == 'nli_proxy' else None,
                'loaded_revision': getattr(model.config, '_commit_hash', None), 'threshold_source': 'original frozen validation' if method == 'original' else 'candidate-v2 validation only',
                'inference_seconds': elapsed, 'tokenization': {'max_pair_tokens': max(lengths), 'truncated_rows': 0},
                'dataset_hash': sha256(rows), 'runtime_versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'huggingface-hub', 'tokenizers']}}
            results[name][method] = report
            write_json(run / 'comparisons' / f'{name}_{method}.json', report)
        del model, tokenizer
        gc.collect()
    for name, rows in datasets.items():
        for method in ('constant', 'identifier'):
            predictions, coverage = baseline_predictions(rows, fitted, method == 'identifier')
            report = report_predictions(rows, predictions, links[name], .5)
            report['baseline_provenance'] = {**fitted, 'coverage': coverage}
            results[name][method] = report
            write_json(run / 'comparisons' / f'{name}_{method}.json', report)
    differences = {}
    for name, methods in results.items():
        original = {p['pair_id']: p for p in methods['original']['predictions']}
        v2 = {p['pair_id']: p for p in methods['candidate_v2']['predictions']}
        differences[name] = {'improvements': [v2[k] for k in original if not original[k]['correct'] and v2[k]['correct']],
                             'regressions': [v2[k] for k in original if original[k]['correct'] and not v2[k]['correct']],
                             'both_wrong': [v2[k] for k in original if not original[k]['correct'] and not v2[k]['correct']]}
    write_json(run / 'differences.json', differences)
    render(run, results, differences, selection)


def render(run, results, differences, selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Candidate v2</title><style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:20px}table{border-collapse:collapse}th,td{padding:8px;border:1px solid #999}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}</style>",
             '<h1>PRELIMINARY / EXPERIMENTAL</h1><p>Provisional synthetic function sensitivity. Stress benchmark informed development. No human review, general accuracy or causal memorization finding. All predictions remain advisory.</p>',
             '<p><a href="protocol.json">Frozen protocol</a> | <a href="manifest.json">Execution attempts</a> | <a href="leakage_audit.json">Leakage audit</a> | <a href="differences.json">Complete improvements/regressions</a></p>']
    history = selection['log_history']
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([h['step'] for h in history if 'loss' in h], [h['loss'] for h in history if 'loss' in h], label='training loss')
    ax.plot([h['step'] for h in history if 'eval_loss' in h], [h['eval_loss'] for h in history if 'eval_loss' in h], 'o-', label='validation loss')
    ax.set(xlabel='actual optimizer steps', ylabel='cross entropy', title='Measured candidate-v2 loss'); ax.legend(); fig.tight_layout()
    fig.savefig(run / 'loss.png'); plt.close(fig)
    lines.append('<h2>Training</h2><img src="loss.png" alt="Measured losses"><pre>' + html.escape(json.dumps({k: selection[k] for k in ['actual_optimizer_steps', 'selected_checkpoint', 'threshold', 'training_metrics', 'checkpoint_sha256']}, indent=2)) + '</pre>')
    for split, methods in results.items():
        lines.append(f'<h2>{split}</h2><table><tr><th>Method</th><th>Accuracy</th><th>Macro F1</th><th>Rename equal / pairs</th><th>Role both correct / pairs</th><th>High-confidence errors</th></tr>')
        for method, r in methods.items():
            a, b = r['paired_metrics']['renaming'], r['paired_metrics']['role_change']
            lines.append(f"<tr><td>{method}</td><td>{r['classification']['accuracy']:.4f}</td><td>{r['macro_f1']:.4f}</td><td>{a['numerator']}/{a['denominator']}</td><td>{b['numerator']}/{b['denominator']}</td><td>{len(r['examples']['high_confidence_incorrect'])}</td></tr>")
        lines.append('</table>')
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for ax, (method, r) in zip(axes, methods.items()):
            ax.imshow(r['confusion_matrix'], cmap='Blues'); ax.set_title(method); ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xlabel('prediction'); ax.set_ylabel('expected')
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, str(r['confusion_matrix'][a][b]), ha='center', va='center')
        fig.tight_layout(); fig.savefig(run / (split + '_confusions.png')); plt.close(fig)
        lines.append(f'<img src="{split}_confusions.png" alt="{split} measured confusion matrices">')
        lines.append('<p>Renaming: equal binary predictions for a validated alias-only transformation (not necessarily correct). Role: both opposite-label counterparts correct. Exact links, row coverage, overlap, probabilities, class supports, per-scenario scores and all high-confidence errors are in each artifact.</p>')
        for method, r in methods.items():
            summary = {k: r[k] for k in ['class_counts', 'confusion_matrix', 'per_class', 'paired_metrics', 'per_scenario']}
            lines.append(f'<details><summary>{method}: per-scenario metrics / pair coverage / error examples</summary><a href="comparisons/{split}_{method}.json">Complete predictions and provenance</a><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre></details>')
        diff = differences[split]
        lines.append(f"<p>Compared with original: {len(diff['improvements'])} improvements, {len(diff['regressions'])} regressions, {len(diff['both_wrong'])} both wrong.</p>")
        lines.append('<pre>' + html.escape(json.dumps({k: v[:3] for k,v in diff.items()}, indent=2)) + '</pre>')
    lines.append('<p>Identifier-only priors were fitted on 256 training rows. Unseen aliases fall back to the training prior. Balanced fallback outcomes do not disprove shortcuts. Final templates and weighted scenario were held out; all results describe this small synthetic distribution only.</p></html>')
    (run / 'report.html').write_text(''.join(lines), encoding='utf-8')
