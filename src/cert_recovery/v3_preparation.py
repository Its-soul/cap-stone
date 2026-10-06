"""Cache-only tokenizer audits and human-review preparation; no model loading."""
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re

from .annotation_review import write_exclusive, V2
from .contrast_data import TEMPLATES, validate_transformations
from .crypto import canonical_json, sha256
from .data_pipeline import clean_rows, read_jsonl
from .dependency_contract import DEFINITION, FORMULAS
from .pinned_smoke import sha256_file

PIN = 'fa2804872c3b4bd748f38c0185cc85775361e735'
MODEL = 'cross-encoder/nli-deberta-v3-small'
ORIGINAL_PROTOCOL_SHA = 'c17cd2158555f878cc6609bd04c360c8ce020cf7849f23e9af9c26986639f9ec'
ROOT_CLAUSE = 'All sources have no supports.'
REPAIRED_DATA_VERSION = 'candidate-v3-data-v2-root-explicit'


def explicit_roots(row):
    """Versioned wording repair, with original bytes/labels retained in parent artifacts."""
    result = copy.deepcopy(row)
    if ROOT_CLAUSE in row['premise'] or 'parent_original_row_sha256' in row:
        raise ValueError('Root clarification already present; do not repair twice')
    for key, marker in [('premise', ' Facts:'), ('hypothesis', ' Check:')]:
        if marker not in result[key]: raise ValueError('Missing frozen observation delimiter')
        result[key] = result[key].replace(marker, ' '+ROOT_CLAUSE+marker, 1)
    result['annotation_version'] = REPAIRED_DATA_VERSION
    result['parent_original_row_sha256'] = sha256(row)
    result['repair'] = 'explicit source root status in both input strings; labels/rules/groups unchanged'
    return result


def original_row(row):
    result = copy.deepcopy(row)
    if 'parent_original_row_sha256' in result:
        parent = result.pop('parent_original_row_sha256')
        if result.pop('annotation_version') != REPAIRED_DATA_VERSION: raise ValueError('Unknown repaired row version')
        if result.pop('repair') != 'explicit source root status in both input strings; labels/rules/groups unchanged':
            raise ValueError('Undeclared repair factor')
        for key, marker in [('premise', ' Facts:'), ('hypothesis', ' Check:')]:
            suffix = ' '+ROOT_CLAUSE+marker
            if result[key].count(suffix) != 1: raise ValueError('Missing/duplicated explicit root contract')
            result[key] = result[key].replace(suffix, marker, 1)
        if sha256(result) != parent: raise ValueError('Repair changed factors beyond root wording')
    return result


def validate_links(rows, links):
    return validate_transformations([original_row(r) for r in rows], links)


def freeze_replacement(out, protocol, rows, links, contrasts):
    target = Path(out)/'replacement'; target.mkdir(exist_ok=False)
    repaired = [explicit_roots(r) for r in rows]
    repaired_contrasts = [explicit_roots(r) for r in contrasts['rows']]
    validate_links(repaired, links); validate_links(repaired_contrasts, contrasts['transformations'])
    for split in ['train', 'validation', 'future_final']:
        with (target/f'{split}.jsonl').open('x', encoding='utf-8') as stream:
            stream.write(''.join(canonical_json(r)+'\n' for r in repaired if r['split'] == split))
    write_exclusive(target/'transformations.json', {'links': links, 'unchanged_from_original': True})
    write_exclusive(target/'contrasts.json', {'version': 'mentor-contrasts-v2-root-explicit', 'rows': repaired_contrasts,
        'transformations': contrasts['transformations'], 'annotations': 'SYNTHETIC_PROVISIONAL', 'human_reviewed': False})
    replacement = copy.deepcopy(protocol)
    replacement.update(version='candidate-v3-plan-v2-root-explicit', dataset_version=REPAIRED_DATA_VERSION,
        frozen_utc=datetime.now(timezone.utc).isoformat(), parent_protocol_sha256=ORIGINAL_PROTOCOL_SHA,
        parent_dataset_file_sha256=protocol['dataset_file_sha256'],
        dataset_file_sha256={s: sha256_file(target/f'{s}.jsonl') for s in ['train', 'validation', 'future_final']},
        transformation_file_sha256=sha256_file(target/'transformations.json'))
    # Parent leakage audit refers to ORIGINAL strings. Explicit constant context creates no new duplicates.
    replacement.pop('leakage_audit_file_sha256')
    replacement['parent_leakage_audit_file_sha256'] = protocol['leakage_audit_file_sha256']
    replacement['repair'] = {'changed_factor': 'example wording: root-source condition formerly in metadata/domain assumptions',
        'added_clause': ROOT_CLAUSE, 'labels_changed': False, 'splits_or_links_changed': False,
        'serialization_changed': False, 'max_length_changed': False, 'model_loss_budget_selection_changed': False,
        'intervention_interpretation': 'additional explicit context within the declared training-distribution intervention; no preprocessing/serialization intervention',
        'original_scope': 'Q direct read list alone cannot rule out a candidate as an ancestor of a derived read support; input does not name ArtifactKind.DEPENDENCY',
        'review_status': 'PROVISIONAL_ROOT_CLARIFICATION_PENDING_HUMAN_REVIEW'}
    exact = lambda r:(r['premise'],r['hypothesis'])
    normalized = lambda r:tuple(model_input(r).values())
    if len({exact(r) for r in repaired}) != len(repaired) or len({normalized(r) for r in repaired}) != len(repaired):
        raise ValueError('Repair introduced duplicate inputs')
    write_exclusive(target/'repair_audit.json', {'status': 'PASS', 'rows': len(repaired),
        'exact_and_normalized_duplicates': 0, 'group_assignments_unchanged': True, 'transformation_links_unchanged': True,
        'parent_structural_template_overlap': 'original overlap audit retained; same constant clause added to every side, no new overlap distinctions',
        'parent_row_hashes': {r['pair_id']: r['parent_original_row_sha256'] for r in repaired}})
    replacement['repair_audit_sha256'] = sha256_file(target/'repair_audit.json')
    write_exclusive(target/'protocol.json', replacement)
    write_exclusive(target/'freeze_manifest.json', {'artifact_sha256': {p.name: sha256_file(p) for p in target.iterdir() if p.is_file()},
        'protocol_sha256': sha256_file(target/'protocol.json'), 'status': 'REPLACEMENT_DISABLED_PENDING_HUMAN'})
    return replacement, repaired, links, {'rows': repaired_contrasts, 'transformations': contrasts['transformations']}


def installed_versions(packages):
    result = {}
    for package in packages:
        try: result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: result[package] = None
    return result


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_proposal(root, review):
    root, review = Path(root), Path(review)
    protocol = load_json(review/'proposal/protocol.json')
    if sha256_file(review/'proposal/protocol.json') != ORIGINAL_PROTOCOL_SHA:
        raise ValueError('This prerequisite plan binds the original disabled v3 protocol')
    for manifest_name, key in [('phase_manifest.json', 'prepared_artifact_hashes'),
                               ('completion_manifest.json', 'artifact_sha256')]:
        for name, digest in load_json(review/manifest_name)[key].items():
            if sha256_file(review/name) != digest:
                raise ValueError('Frozen proposal artifact changed: '+name)
    for name, digest in load_json(review/'phase_manifest.json')['preparation_source_hashes'].items():
        if sha256_file(root/name) != digest:
            raise ValueError('Frozen preparation source changed: '+name)
    if any(protocol['execution'].values()):
        raise ValueError('Original proposal must remain disabled')
    rows = [r for split in ['train', 'validation', 'future_final']
            for r in read_jsonl(review/'proposal'/f'{split}.jsonl')]
    links = load_json(review/'proposal/transformations.json')['links']
    indexed = validate_transformations(rows, links)
    if len(indexed) != 448 or {r['group_id']: r['split'] for r in rows} != protocol['group_assignments']:
        raise ValueError('Frozen split assignments differ')
    if any('prediction' in r or r['human_reviewed'] for r in rows):
        raise ValueError('Proposed rows must have no predictions or fabricated review')
    return protocol, rows, links


def model_input(row):
    """Exactly the existing cleaner and model pipeline's two text fields."""
    cleaned, _ = clean_rows([row], 'world_id')
    return {k: cleaned[0][k] for k in ['premise', 'hypothesis']}


def requirements(row):
    """Expected decisive text from frozen factors; absence is a failure, not metadata rescue."""
    f = row['fields']; a, b = f['supports']; template = f['template']
    support = [f'Q in {f["world"]} reads ONLY {a} and {b}',
               f'Q declares exactly {a}, {b} as read supports',
               f"Q's exhaustive read support is {a} and {b}"][template]
    query = [f'candidate: {f["candidate"]}', f'source queried: {f["candidate"]}',
             f'candidate {f["candidate"]}'][template]
    scope = ['Observation check is separate from Q support.', 'Check concerns observations only.',
             'Separate observation check follows.'][template]
    common = {'named_target_and_supports': support, 'target_scope': scope,
              'support_rule': FORMULAS[f['rule_kind']], 'exhaustivity': ['ONLY', 'exactly', 'exhaustive'][template],
              'queried_source': query}
    # Keep any decisive negation/zero qualifier inside the entire rule span.
    common['negation_or_zero_qualifier'] = FORMULAS[f['rule_kind']] if f['rule_kind'] in {'zero_read', 'cancellation'} else None
    # Known direct membership proves positives; negatives additionally require closed ancestry.
    common['source_ancestry_contract'] = ROOT_CLAUSE if row['label'] == 0 else None
    return {side: {**common, 'observation_evidence' if side == 'premise' else 'observation_check':
                   row['nli_projection'][side]} for side in ['premise', 'hypothesis']}


def span_retained(full, kept, sequence, start, end):
    def tokens(encoding):
        return Counter((left, right, token) for token, (left, right), seq in
                       zip(encoding['input_ids'], encoding['offset_mapping'], encoding.sequence_ids())
                       if seq == sequence and right > left and right > start and left < end)
    original, retained = tokens(full), tokens(kept)
    return bool(original) and not (original - retained)


def audit_row(row, tokenizer, max_length=128):
    pair = model_input(row)
    full = tokenizer(pair['premise'], pair['hypothesis'], truncation=False,
                     return_offsets_mapping=True, return_special_tokens_mask=True)
    kept = tokenizer(pair['premise'], pair['hypothesis'], truncation=True, max_length=max_length,
                     return_offsets_mapping=True, return_special_tokens_mask=True)
    clauses = []; missing = []; lost = []
    for sequence, side in enumerate(['premise', 'hypothesis']):
        for kind, text in requirements(row)[side].items():
            if text is None:
                clauses.append({'side': side, 'kind': kind, 'status': 'NOT_APPLICABLE'})
                continue
            start = pair[side].find(text)
            end = start + len(text) if start >= 0 else None
            status = ('MISSING_FROM_MODEL_TEXT' if start < 0 else 'RETAINED'
                      if span_retained(full, kept, sequence, start, end) else 'LOST_DURING_ENCODING')
            clauses.append({'side': side, 'kind': kind, 'text': text, 'character_span': [start, end], 'status': status})
            if status == 'MISSING_FROM_MODEL_TEXT': missing.append(side+':'+kind)
            if status == 'LOST_DURING_ENCODING': lost.append(side+':'+kind)
    return {'case_id': row['pair_id'], 'split': row['split'], 'scenario': row['scenario'],
            'input': pair, 'input_sha256': sha256(pair), 'original_row_sha256': sha256(row),
            'original_length_including_special_tokens': len(full['input_ids']),
            'retained_length_including_special_tokens': len(kept['input_ids']),
            'special_token_count': sum(kept['special_tokens_mask']),
            'truncated': len(full['input_ids']) > len(kept['input_ids']),
            'clauses': clauses, 'missing_information': missing, 'lost_information': lost,
            'essential_information_retained': not missing and not lost,
            'input_ids': kept['input_ids'], 'attention_mask': kept['attention_mask'],
            'sequence_ids': kept.sequence_ids(), 'offset_mapping': kept['offset_mapping']}


def audit_transformations(rows, links, audits):
    validate_links(rows, links)
    indexed = {r['case_id']: r for r in audits}
    if len(indexed) != len(audits) or set(indexed) != {r['pair_id'] for r in rows}:
        raise ValueError('Encoded audits must align with all exact rows')
    result = []
    for link in links:
        a, b = [indexed[link[k]] for k in ['parent_pair_id', 'child_pair_id']]
        reasons = []
        for item in [a, b]:
            if not item['essential_information_retained']:
                reasons.append(item['case_id']+': decisive information missing/lost')
        if a['input_ids'] == b['input_ids']:
            reasons.append('Declared changed factor collapsed to identical encoded input; opposite-label collision' if
                           link['kind'] == 'support_rule' else 'Declared changed factor is invisible after encoding')
        result.append({'transformation_id': link['transformation_id'], 'kind': link['kind'],
                       'parent_pair_id': a['case_id'], 'child_pair_id': b['case_id'],
                       'status': 'INVALID_ENCODED_PAIR' if reasons else 'VALID', 'reasons': reasons,
                       'expected_dependency_labels': link['expected_dependency_labels'],
                       'expected_nli_labels': link['expected_nli_labels']})
    return result


def audit_collection(rows, links, tokenizer, max_length=128):
    audits = [audit_row(r, tokenizer, max_length) for r in rows]
    transformations = audit_transformations(rows, links, audits)
    counts = {}
    for split in sorted({r['split'] for r in rows}):
        selected = [r for r in audits if r['split'] == split]
        counts[split] = {}
        for scenario in sorted({r['scenario'] for r in selected}):
            group = [r for r in selected if r['scenario'] == scenario]
            counts[split][scenario] = {'rows': len(group), 'truncated_rows': sum(r['truncated'] for r in group),
                'max_original_tokens': max(r['original_length_including_special_tokens'] for r in group),
                'missing_information_rows': sum(bool(r['missing_information']) for r in group),
                'lost_information_rows': sum(bool(r['lost_information']) for r in group)}
    failed = sum(not r['essential_information_retained'] for r in audits)
    invalid = sum(r['status'] != 'VALID' for r in transformations)
    truncated = sum(r['truncated'] for r in audits)
    return {'status': 'PASS' if not failed and not invalid and not truncated else 'REPAIR_REQUIRED',
            'configuration': {'max_length': max_length, 'truncation': 'longest_first (truncation=True)',
                'truncation_side': tokenizer.truncation_side, 'padding_side': tokenizer.padding_side,
                'training_padding': 'dynamic DataCollatorWithPadding', 'inference_padding': 'padding=True per batch',
                'special_tokens_included': True, 'input_order': 'premise_then_hypothesis',
                'audit_padding': 'unpadded content lengths; batch-padding equivalence checked separately'},
            'rows': audits, 'transformations': transformations, 'counts_by_split_scenario': counts,
            'truncated_rows': truncated, 'decisive_information_failures': failed, 'invalid_encoded_pairs': invalid,
            'max_original_tokens': max(r['original_length_including_special_tokens'] for r in audits),
            'interpretation': 'Token retention and explicit visible contract checks; not human annotation approval or model quality.'}


def local_tokenizer(root):
    root = Path(root)
    cache = root/'_private/runtime/hf-cache/hub'
    snapshot = cache/'models--cross-encoder--nli-deberta-v3-small/snapshots'/PIN
    files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'spm.model', 'config.json']
    missing = [str(snapshot/name) for name in files if not (snapshot/name).is_file()]
    if missing:
        raise FileNotFoundError('Missing pinned local tokenizer files; downloads prohibited: '+', '.join(missing))
    if importlib.metadata.version('transformers') != '4.57.1':
        raise ValueError('Exact proposed Transformers 4.57.1 runtime required for tokenizer audit')
    from transformers import AutoTokenizer  # Tokenizer only. No AutoModel/weight loading.
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=PIN, cache_dir=str(cache),
        local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.truncation_side != 'right':
        raise ValueError('Expected fast pinned tokenizer with right truncation and offset mappings')
    provenance = {'model_id': MODEL, 'revision': PIN, 'snapshot': str(snapshot),
        'tokenizer_class': type(tokenizer).__name__, 'is_fast': tokenizer.is_fast,
        'files': {name: sha256_file(snapshot/name) for name in files},
        'backend_sha256': sha256(tokenizer.backend_tokenizer.to_str()),
        'runtime_versions': {p: importlib.metadata.version(p) for p in ['transformers', 'tokenizers', 'sentencepiece', 'huggingface-hub']},
        'special_tokens': tokenizer.special_tokens_map, 'pair_special_token_count': tokenizer.num_special_tokens_to_add(pair=True),
        'loading': 'AutoTokenizer.from_pretrained(model ID, immutable revision, local_files_only=True); no weights loaded'}
    return tokenizer, provenance


def check_batch_equivalence(rows, tokenizer, audits, batch_size=2, max_length=128):
    index = {r['case_id']: r for r in audits}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]; pairs = [model_input(r) for r in batch]
        encoding = tokenizer([r['premise'] for r in pairs], [r['hypothesis'] for r in pairs],
                             padding=True, truncation=True, max_length=max_length)
        for row, ids, mask in zip(batch, encoding['input_ids'], encoding['attention_mask']):
            retained = [token for token, attended in zip(ids, mask) if attended]
            if retained != index[row['pair_id']]['input_ids']:
                raise ValueError('Actual batch tokenization differs from audited row: '+row['pair_id'])
    return {'status': 'PASS', 'batch_size': batch_size, 'rows_checked': len(rows),
            'configuration': 'padding=True, truncation=True, max_length=128; same content as dynamic training padding'}


def prepare_splits(out, rows, protocol):
    cleaned, cleaning = clean_rows(rows, 'world_id')
    if len(cleaned) != len(rows):
        raise ValueError('Unexpected deduplication of frozen candidates')
    indexed = {r['pair_id']: r for r in rows}
    splits = {name: [] for name in ['train', 'validation', 'test']}
    for row in cleaned:
        original = indexed[row['pair_id']]
        if model_input(original) != {k: row[k] for k in ['premise', 'hypothesis']}:
            raise ValueError('Model-visible construction differs from existing cleaner')
        name = original['split']; name = 'test' if name == 'future_final' else name
        splits[name].append(row)
    target = Path(out)/'prepared_splits'; target.mkdir(exist_ok=False)
    for name, selected in splits.items():
        with (target/f'{name}.jsonl').open('x', encoding='utf-8') as stream:
            stream.write(''.join(canonical_json(r)+'\n' for r in selected))
    manifest = {'stage': 'OFFLINE_V3_PREREQUISITES', 'cleaning': cleaning, 'seed': 42, 'group_key': 'world_id',
        'fractions': [len(splits[n])/len(rows) for n in ['train', 'validation', 'test']],
        'assignment_policy': 'original frozen group_assignments; future_final aliases test for existing loader only',
        'normalized_dataset_hash': sha256(cleaned),
        'splits': {name: {'rows': len(items), 'hash': sha256(items), 'groups': sorted({r['world_id'] for r in items})}
                   for name, items in splits.items()}}
    write_exclusive(target/'split_manifest.json', manifest)
    return {'manifest_sha256': sha256_file(target/'split_manifest.json'),
            'file_sha256': {n: sha256_file(target/f'{n}.jsonl') for n in splits}}


def review_item(row, artifact, version, question, historical=False):
    pair = row['input'] if historical else model_input(row)
    if historical:
        evidence = {'review_rationale': row['explanation'], 'original_derivation': row['original_derivation'],
                    'scope': row['target_scope']}
    else:
        d = row['derivation']; w = d['intervention']
        evidence = {k: d[k] for k in ['target', 'declared_support_graph', 'complete_nodes', 'rule', 'label_basis']}
        evidence['intervention'] = {k: w[k] for k in ['before', 'after', 'value_changed', 'changes',
            'old_certificate_valid_after_change', 'new_certificate_version', 'current_certificate_valid', 'declared_supports']}
    return {'item_id': row['review_id'] if historical else row['pair_id'], 'model_visible_input': pair,
        'proposed_dependency_label': row['original_label'] if historical else row['label'],
        'proposed_nli_label': row['semantic_relation'] if historical else row['nli_label'],
        'rule_evidence': evidence,
        **({'observed_saved_predictions': row['predictions'], 'finding': row['finding']} if historical else {}),
        'reviewer_question': question, 'reviewed_artifact_version': {'artifact': artifact, 'version': version,
            'row_sha256': row['original_row_sha256'] if historical else sha256(row), 'input_sha256': sha256(pair)},
        'reviewer_identity': None, 'decision': None, 'rationale': None, 'unresolved_ambiguity': None,
        'human_review_status': 'PENDING'}


def mentor_packet(review, rows, links, contrast_bundle=None, protocol_sha=ORIGINAL_PROTOCOL_SHA):
    review = Path(review); contrast_bundle = contrast_bundle or load_json(review/'contrasts.json')
    contrasts = contrast_bundle['rows']; records = load_json(review/'review.json')['records']
    repaired = 'parent_original_row_sha256' in rows[0]
    questions = 'Does the text explicitly make Q the target and this candidate a declared support (1), or exhaustively exclude it (0), with read sources having no supports, independently of the separate NLI check?'
    contrast_items = [review_item(r, 'replacement/contrasts.json' if repaired else 'contrasts.json',
        'mentor-contrasts-v2-root-explicit' if repaired else 'mentor-contrasts-v1', questions) for r in contrasts]
    final_items = [review_item(r, 'replacement/future_final.jsonl' if repaired else 'proposal/future_final.jsonl',
        REPAIRED_DATA_VERSION if repaired else 'candidate-v3-data-v1', questions) for r in rows if r['split'] == 'future_final']
    historical = [r for r in records if r['finding'] == 'MODEL_ERROR_UNDER_EXPLICIT_PROVISIONAL_RULE']
    for scenario in ['nli_context', 'approval', 'nli_trap']:
        historical.append(next(r for r in records if r['scenario'] == scenario and not r['text_sufficient_without_scope_assumptions']))
    historical_items = [review_item(r, 'review.json', 'retrospective-review-v1',
        'Is the recorded conclusion justified under the stated scope, or missing target/exhaustivity? Preserve the old label and give a separate rationale.', True) for r in historical]
    highlight_ids = []
    for relation in ['entailment', 'contradiction', 'neutral']:
        for label in [0, 1]:
            highlight_ids.append(next(r['pair_id'] for r in contrasts if r['nli_label'] == relation and r['label'] == label))
    for scenario in ['zero_read', 'cancellation', 'redundant_or']:
        highlight_ids.append(next(r['pair_id'] for r in contrasts if r['scenario'] == scenario and r['label'] == 1))
    control_links = [next(l for l in contrast_bundle['transformations'] if l['kind'] == kind)
                     for kind in ['renaming', 'support_rule']]
    highlight_ids = list(dict.fromkeys(highlight_ids + [l[k] for l in control_links for k in ['parent_pair_id', 'child_pair_id']]))
    contract = {'definition': DEFINITION, 'target': 'Q only, separate observation check',
        'reviewed_artifact_version': {'protocol_sha256': protocol_sha, 'contract_version': 'declared-support-closure-v1'},
        'reviewer_identity': None, 'decision': None, 'rationale': None, 'unresolved_ambiguity': None, 'human_review_status': 'PENDING'}
    return {'status': 'PENDING_HUMAN_REVIEW', 'human_reviewed': False, 'genuine_feedback_available': False,
        'contract_review': contract, 'highlight_item_ids': highlight_ids, 'controlled_examples': control_links,
        'contrast_reviews': contrast_items, 'future_final_reviews': final_items, 'historical_examples': historical_items,
        'required_review': 'Approve target/scope contract, all 64 contrasts and all 128 future-final labels in NEW external human review sidecar; any unresolved ambiguity blocks launch.',
        'note': 'Blank reviewer fields are intentional. Automated preparation is not human approval. Historical examples are retrospective, not replacement labels.'}


def write_packet(out, packet):
    out = Path(out); write_exclusive(out/'mentor_packet.json', packet)
    for name, items in [('mentor_contrasts.csv', packet['contrast_reviews']), ('mentor_future_final.csv', packet['future_final_reviews']),
                        ('mentor_historical.csv', packet['historical_examples'])]:
        with (out/name).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['item_id', 'premise', 'hypothesis', 'dependency_label', 'NLI_label', 'rule_evidence',
                             'reviewer_question', 'reviewed_artifact_version', 'reviewer_identity', 'decision', 'rationale', 'unresolved_ambiguity'])
            for item in items:
                writer.writerow([item['item_id'], item['model_visible_input']['premise'], item['model_visible_input']['hypothesis'],
                    item['proposed_dependency_label'], item['proposed_nli_label'], canonical_json(item['rule_evidence']),
                    item['reviewer_question'], canonical_json(item['reviewed_artifact_version']), '', '', '', ''])
    lines = ['# Mentor review packet', '', 'PENDING HUMAN REVIEW. No mentor feedback was supplied or fabricated.', '',
        DEFINITION, '', 'Review Q support separately from the observation check. Check every label 0 against an exhaustive contract.', '',
        'The JSON and CSV sheets contain exact model-visible inputs, rule evidence and reviewer fields. All 64 contrasts and 128 future-final rows require review.', '',
        '## Compact examples', '']
    highlighted = {i['item_id'] for i in packet['contrast_reviews'] if i['item_id'] in packet['highlight_item_ids']}
    for item in [i for i in packet['contrast_reviews'] if i['item_id'] in highlighted] + packet['historical_examples']:
        lines.extend(['### '+item['item_id'], '', 'Premise: '+item['model_visible_input']['premise'], '',
            'Hypothesis: '+item['model_visible_input']['hypothesis'], '',
            f"Provisional dependency: {item['proposed_dependency_label']}; NLI: {item['proposed_nli_label']}.", '',
            'Rule evidence: '+canonical_json(item['rule_evidence']), '', 'Review question: '+item['reviewer_question'], ''])
    lines += ['## Controlled transformations', '', canonical_json(packet['controlled_examples']), '',
        'Record decisions in a new approval sidecar based on approval.template.json. Include reviewer identity, UTC date, rationale, unresolved ambiguity, exact artifact bindings and every required item. Preserve the templates and historical labels.', '']
    with (out/'MENTOR_PACKET.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))


def interpretation_note(review):
    review = Path(review); records = load_json(review/'review.json')['records']
    subsets = {'explicit_provisional_rules': [r for r in records if r['finding'] in ['CONSISTENT_PROVISIONAL_CONTRACT', 'MODEL_ERROR_UNDER_EXPLICIT_PROVISIONAL_RULE']],
               'scope_dependent': [r for r in records if r['finding'] == 'SCOPE_AMBIGUITY_CONDITIONAL_MODEL_ERROR'],
               'underdetermined': [r for r in records if r['finding'] == 'UNDERDETERMINED_ANNOTATION']}
    diagnostics = {}
    for name, selected in subsets.items():
        diagnostics[name] = {'inclusion_rule': 'review.finding membership; see case IDs, original labels unchanged',
            'findings': sorted({r['finding'] for r in selected}), 'case_ids': [r['review_id'] for r in selected],
            'denominator': len(selected), 'full_review_denominator': len(records), 'coverage': len(selected)/len(records),
            'correct_against_original_label': {method: {'numerator': sum(r['predictions'][method]['prediction'] == r['original_label'] for r in selected),
                'denominator': len(selected)} for method in ['original', 'candidate_v2']},
            'warning': 'RETROSPECTIVE DIAGNOSTIC; counts against original provisional labels. Under-determined/scope-dependent agreement is not semantic correctness.'}
    return {'status': 'ADDITIVE_RETROSPECTIVE_INTERPRETATION', 'source_review_sha256': sha256_file(review/'review.json'),
        'source_corrections_sha256': sha256_file(review/'corrections.json'), 'original_scores_preserved': True,
        'confirmed_under_explicit_provisional_rules': [r['review_id'] for r in subsets['explicit_provisional_rules'] if r['v2_mismatch_against_original_label']],
        'scope_dependent_conclusions': 16, 'underdetermined_original_labels': 128,
        'invalid_transformation_counts': load_json(review/'historical_transformations.json')['counts'],
        'subset_diagnostics': diagnostics, 'no_revised_headline_score': True,
        'corrections': ['Eight high-score validation mismatches are part of the 16 Q-scope cases, not eight extra errors.',
            'All 49 old-label stress regressions lack declared target/support context; no binary inversion established.',
            '42 old-label gains comprise 22 arithmetic and 20 approval; approval absence relies on unstated exhaustivity.',
            '189 rename proposals change facts; 191 role counterparts change more than one factor.',
            'V2 two-member support-set swaps are not isolated single-support substitutions.',
            'No causal memorization or calibrated confidence conclusion is established.']}


def prepare_bundle(root, review, out, hardware):
    root, review, out = Path(root).resolve(), Path(review).resolve(), Path(out).resolve()
    protocol, rows, links = verify_proposal(root, review)
    contrasts = load_json(review/'contrasts.json')
    original_all_rows = rows + contrasts['rows']; original_all_links = links + contrasts['transformations']
    protocol, rows, links, contrasts = freeze_replacement(out, protocol, rows, links, contrasts)
    all_rows = rows + contrasts['rows']; all_links = links + contrasts['transformations']
    audit = {'status': 'BLOCKED', 'blocker': None, 'model_loading': False, 'inference': False}
    try:
        tokenizer, provenance = local_tokenizer(root)
    except (FileNotFoundError, ImportError, importlib.metadata.PackageNotFoundError, ValueError) as error:
        audit['blocker'] = {'type': type(error).__name__, 'message': str(error)}
    else:
        original_audit = audit_collection(original_all_rows, original_all_links, tokenizer, protocol['controls']['starting_model']['max_length'])
        original_audit['tokenizer_provenance'] = provenance
        write_exclusive(out/'original_tokenizer_and_context_audit.json', original_audit)
        audit = audit_collection(all_rows, all_links, tokenizer, protocol['controls']['starting_model']['max_length'])
        audit['tokenizer_provenance'] = provenance
        audit['batch_equivalence'] = check_batch_equivalence(all_rows, tokenizer, audit['rows'])
        from transformers import AutoTokenizer
        comparator_tokenizer = AutoTokenizer.from_pretrained(root/V2/'checkpoints/best', local_files_only=True,
                                                             trust_remote_code=False)
        different = []
        for row in all_rows:
            pair = model_input(row)
            a = tokenizer(pair['premise'], pair['hypothesis'], truncation=True, max_length=128)['input_ids']
            b = comparator_tokenizer(pair['premise'], pair['hypothesis'], truncation=True, max_length=128)['input_ids']
            if a != b: different.append(row['pair_id'])
        audit['comparator_tokenizer_equivalence'] = {'rows_checked': len(all_rows), 'different_rows': different,
            'status': 'PASS' if not different else 'BLOCKED',
            'comparator_files': {p.name: sha256_file(p) for p in (root/V2/'checkpoints/best').iterdir()
                                 if p.is_file() and p.name in provenance['files']}}
        if different: audit['status'] = 'REPAIR_REQUIRED'
    write_exclusive(out/'tokenizer_audit.json', audit)
    input_audit = {'status': 'PASS' if audit.get('decisive_information_failures') == 0 else 'BLOCKED',
        'construction': 'clean_rows normalizes NFC/whitespace; model_pipeline.tokenize and _infer pass ONLY premise, hypothesis',
        'same_as_v2': True, 'metadata_appended_to_input': False, 'nli_projection_used_as_model_input': False,
        'input_constructor_changed': False, 'max_length_changed': False, 'serialization_intervention': False,
        'dataset_text_intervention': 'versioned root-source clarification added to both sides; original proposal retained',
        'essential_information_only_in_metadata': [r['case_id'] for r in audit.get('rows', []) if r['missing_information']],
        'visible_contract_scope': 'named Q, exhaustive declared read supports, operation, queried source and separate observation check',
        'source_hashes': {p: sha256_file(root/p) for p in ['src/cert_recovery/data_pipeline.py', 'src/cert_recovery/model_pipeline.py']}}
    write_exclusive(out/'input_construction.json', input_audit)
    splits = prepare_splits(out, rows, protocol)
    protocol_path = out/'replacement/protocol.json' if (out/'replacement/protocol.json').exists() else review/'proposal/protocol.json'
    protocol_sha = sha256_file(protocol_path)
    packet = mentor_packet(review, rows, links, contrasts, protocol_sha); write_packet(out, packet)
    approval = copy.deepcopy(packet)
    approval.update({'record_kind': 'EXTERNAL_HUMAN_REVIEW_REQUIRED', 'reviewed_at_utc': None,
                     'reviewed_protocol_sha256': protocol_sha,
                     'reviewed_dataset_sha256': protocol['dataset_file_sha256']})
    write_exclusive(out/'approval.template.json', approval)
    write_exclusive(out/'historical_interpretation.json', interpretation_note(review))
    with (out/'HISTORICAL_INTERPRETATION.md').open('x', encoding='utf-8') as stream:
        stream.write('# Additive retrospective interpretation\n\nOriginal datasets, labels and headline scores remain unchanged.\n'
            'Two arithmetic false positives are confirmed under explicit provisional rules. Sixteen nli_context conclusions require Q-only scope.\n'
            '128 negative labels are underdetermined, including every one of the 49 measured regressions.\n'
            '189 purported renames and 191 role counterparts are invalid as single-factor comparisons.\n'
            'The 20 approval improvements among 42 gains are conditional on unstated exhaustivity.\n'
            'historical_interpretation.json reports retrospective subsets with exact inclusion rules, case IDs, numerator, denominator and coverage; none replaces an original score.\n'
            'High softmax is not calibrated confidence; no causal memorization or generalization claim follows.\n')
    selection = load_json(root/V2/'checkpoints/selection.json')
    comparator = root/V2/'checkpoints/best/model.safetensors'
    if sha256_file(comparator) != selection['checkpoint_sha256']:
        raise ValueError('Frozen v2 comparator weights changed')
    cache_base = root/'_private/runtime/hf-cache/hub/models--cross-encoder--nli-deberta-v3-small/snapshots'/PIN/'model.safetensors'
    blockers = [] if audit['status'] == 'PASS' else ['tokenizer audit: '+audit['status']]
    if not cache_base.is_file(): blockers.append('pinned base weights absent locally; downloads prohibited')
    elif sha256_file(cache_base) != selection['loaded_provenance']['base_weight_sha256']:
        blockers.append('pinned base weight digest differs from recorded v2 base')
    launch = {'schema_version': 1, 'version': 'candidate-v3-launch-v1', 'status': 'DISABLED_PENDING_HUMAN_REVIEW_AND_AUTHORIZATION',
        'approved_dataset_protocol_versions': None, 'proposed_dataset_version': protocol['dataset_version'],
        'proposed_protocol_version': protocol['version'], 'review_dir': str(review), 'protocol_sha256': protocol_sha,
        'protocol_path': str(protocol_path), 'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA,
        'dataset_file_sha256': protocol['dataset_file_sha256'], 'group_assignments': protocol['group_assignments'],
        'prepared_splits': splits, 'prepared_splits_directory': str(out/'prepared_splits'),
        'input_construction_changed': False, 'repairs': [protocol['repair']] if 'repair' in protocol else [],
        'additional_experimental_interventions': [],
        'controls': protocol['controls'], 'selection': protocol['selection'], 'acceptance_criteria': protocol['acceptance_criteria'],
        'evaluation': protocol['evaluation'], 'execution_enabled': False, 'human_review_status': 'PENDING',
        'tokenizer_audit_sha256': sha256_file(out/'tokenizer_audit.json'), 'tokenizer_gate': audit['status'],
        'mentor_packet_sha256': sha256_file(out/'mentor_packet.json'), 'technical_blockers': blockers,
        'pinned_base': {'path': str(cache_base), 'sha256': selection['loaded_provenance']['base_weight_sha256']},
        'frozen_v2_comparator': {'directory': str(comparator.parent), 'weights_sha256': selection['checkpoint_sha256'],
            'selection_sha256': sha256_file(root/V2/'checkpoints/selection.json'), 'threshold': selection['threshold'],
            'tokenizer_hashes': audit.get('comparator_tokenizer_equivalence', {}).get('comparator_files', {})},
        'future_final_access': 'structural/tokenizer/human label review allowed; model predictions only in comparison after successful training and immutable validation selection freeze',
        'hardware': hardware, 'bounded_runtime': {'training_attempts': 1, 'epochs': 2, 'max_optimizer_steps': 256,
            'training_timeout_seconds': 1800, 'comparison_timeout_seconds': 1800, 'device': 'CPU', 'torch_threads': 4,
            'paid_compute': False, 'automatic_retries': False},
        'runtime_versions': installed_versions(['transformers', 'torch', 'tokenizers', 'datasets', 'accelerate', 'huggingface-hub']),
        'attempt_policy': 'new caller-named attempt directory; one exclusive launch claim per preparation bundle; preserve errors/timeouts and never retry automatically',
        'authority': protocol['authority']}
    write_exclusive(out/'launch_spec.json', launch)
    write_exclusive(out/'authorization.template.json', {'training_authorized': False, 'inference_authorized': False,
        'future_final_inference_authorized': False, 'authorized_by': None, 'authorized_at_utc': None,
        'launch_spec_sha256': sha256_file(out/'launch_spec.json'), 'max_training_attempts': 1,
        'note': 'Future external authorization required. Preserve this template; record authorization in a NEW file.'})
    return launch
