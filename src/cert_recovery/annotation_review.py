"""Saved-evidence review and disabled experiment preparation; no model execution."""
from collections import Counter
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import statistics

from .contrast_data import generate_collection, generate_v3_candidates, audit_candidates
from .crypto import canonical_json, sha256
from .data_pipeline import read_jsonl
from .dependency_contract import CONTRACT_VERSION, DEFINITION
from .paired_metrics import align_predictions
from .pinned_smoke import sha256_file

V2 = '_private/runs/candidate_v2_20261005T160614_302922Z_906cc116'
AUDIT = '_private/runs/evidence_audit_20261005T160017_332679Z_466fdd06'
STRESS = '_private/runs/eval_robust_20261005T111536_633570Z/dependency_pairs.jsonl'


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def validate_sidecar(review_path, corrections_path):
    """Check exact hash-bound proposals; never apply them to original labels."""
    review_path, corrections_path = Path(review_path), Path(corrections_path)
    review = json.loads(review_path.read_text(encoding='utf-8'))
    sidecar = json.loads(corrections_path.read_text(encoding='utf-8'))
    if sidecar['original_review_sha256'] != sha256_file(review_path):
        raise ValueError('Sidecar no longer binds the original review bytes')
    expected = [{'review_id': r['review_id'], 'original_row_sha256': r['original_row_sha256'],
                 'original_label': r['original_label'], 'proposed': r['proposed_correction'],
                 'status': 'PENDING_HUMAN_REVIEW', 'human_reviewed': False}
                for r in review['records'] if r['proposed_correction']]
    if sidecar['proposals'] != expected or len({r['review_id'] for r in expected}) != len(expected):
        raise ValueError('Changed/duplicate/unbound correction or invented human approval')
    return len(expected)


def review_example(row, original, v2, original_threshold, v2_threshold, split):
    if any(p[k] != row[k] for p in [original,v2] for k in ['pair_id','premise','hypothesis','label']):
        raise ValueError('Review prediction/input mismatch')
    scenario = row.get('scenario') or row['template_family']
    d = row['derivation']; candidate = d['intervention']['source']
    status, sufficient, proposed, explanation = 'CONSISTENT_PROVISIONAL_CONTRACT',True,{},''
    semantic_relation = 'neutral_under_literal_facts'
    supports = d.get('required_sources')
    if scenario == 'arithmetic':
        supports = row['hypothesis'].split('from ',1)[1].rstrip('.').split(' and ')
        explanation = 'Computed exactly from the named inputs is exhaustive for this task; source membership gives the original label.'
    elif scenario == 'approval':
        supports = row['hypothesis'].split('from ',1)[1].rstrip('.').split(' and ')
        explanation = 'Named sign-off establishes positive support; it does not exclude other required signatories.'
        if row['label'] == 0:
            status,sufficient = 'UNDERDETERMINED_ANNOTATION',False
            proposed = {'binary_label':None,'action':'exclude until an exhaustive support contract is supplied; do not invert the old label'}
    elif scenario == 'nli_trap':
        supports = [candidate] if row['label'] else None
        explanation = ('The hypothesis explicitly declares gateway dependency for the positive.' if row['label'] else
            'No target artifact or exhaustive support rule is given. Operational/identical data is not implied by being backup storage. Dependency cannot be determined; the generator entailment narrative is unsupported.')
        if row['label']==0:
            status,sufficient = 'UNDERDETERMINED_ANNOTATION',False
            proposed={'binary_label':None,'nli_interpretation':'neutral under stated facts, provisional',
                      'action':'exclude until target and support rule supplied; no evidence for a 0-to-1 label flip'}
    else:
        if supports is None: raise ValueError('Missing reviewed support derivation')
        explanation='Synthetic declared target-rule supports and the recorded witness agree with source membership. This remains provisional, not human-reviewed.'
        if scenario == 'nli_context':
            semantic_relation='entailment_under_task_output_scope_provisional'
            explanation=('Target Q is the computed numeric task outcome under the saved derivation. The candidate is excluded for label 0. '
                         'The hypothesis also repeats a fact about that source; certifying the WHOLE hypothesis would require that fact. '
                         'Without explicit target scoping, the input admits different dependency questions. Under Q-only scope, the wrong prediction is a false dependency alarm.')
            if row['label']==0:
                status='SCOPE_AMBIGUITY_CONDITIONAL_MODEL_ERROR';sufficient=False
                proposed={'binary_label':row['label'],'target_scope':'Q computed outcome only, not the complete hypothesis',
                          'action':'retain original label under saved derivation; clarify scope in NEW inputs only'}
    derived_label = int(candidate in supports) if supports is not None else None
    # Non-exhaustive historical negative sign-off does not establish absence.
    if scenario=='approval' and row['label']==0: derived_label=None
    if sufficient and derived_label != row['label']:
        raise ValueError('Original annotation contradicts reviewed complete support rule')
    if sufficient and v2['prediction'] != row['label']:
        status='MODEL_ERROR_UNDER_EXPLICIT_PROVISIONAL_RULE'
    predictions = {}
    for name,p,threshold in [('original',original,original_threshold),('candidate_v2',v2,v2_threshold)]:
        if int(p['probability']>=threshold)!=p['prediction']: raise ValueError('Saved threshold mismatch')
        predictions[name]={'prediction':p['prediction'],'probability':p['probability'],'threshold':threshold,
                           'predicted_label_probability':p['predicted_label_probability'],
                           'confidence_origin':'captured_model_softmax_full_precision', 'calibrated':False,
                           'raw_logits':p['raw_logits'],'model_class_probabilities':p['original_model_class_probabilities']}
    outcome = ('regression' if original['correct'] and not v2['correct'] else 'improvement'
               if not original['correct'] and v2['correct'] else 'both_wrong' if not v2['correct'] else 'both_correct')
    return {'review_id':split+':'+row['pair_id'],'case_id':row['pair_id'],'split':split,'scenario':scenario,
            'input':{'premise':row['premise'],'hypothesis':row['hypothesis']}, 'original_label':row['label'],
            'original_row_sha256':sha256(row), 'declared_rule':d.get('rule'),'original_derivation':d,
            'candidate_source':candidate,'target_scope':'computed task outcome Q; whole hypothesis is a different obligation',
            'inferred_declared_supports':supports,'reviewed_dependency_label_under_explicit_scope':derived_label,
            'text_sufficient_without_scope_assumptions':sufficient,'semantic_relation':semantic_relation,
            'predictions':predictions,'observed_outcome_against_original_annotation':outcome,
            'v2_mismatch_against_original_label':not v2['correct'],
            'v2_high_confidence_mismatch':not v2['correct'] and v2['predicted_label_probability']>=.9,
            'finding':status,'explanation':explanation,'proposed_correction':proposed,
            'review_status':'AUTOMATED_PROVISIONAL_REVIEW_PENDING_MENTOR','human_reviewed':False,
            'observations_not_causation':'No causal memorization conclusion; lexical/objective/scope/optimization mechanisms remain hypotheses.'}


def collect_reviews(root):
    root=Path(root);run=root/V2
    raw=read_jsonl(run/'dependency_pairs.jsonl')
    datasets={'validation':[r for r in raw if r['split']=='validation'], 'final':[r for r in raw if r['split']=='test'],
              'stress':read_jsonl(root/STRESS)}
    records=[];source_files={}
    for split,rows in datasets.items():
        paths={method:run/'comparisons'/f'{split}_{method}.json' for method in ['original','candidate_v2']}
        data={method:json.loads(p.read_text(encoding='utf-8')) for method,p in paths.items()}
        indexed={method:align_predictions(rows,d['predictions'])[1] for method,d in data.items()}
        source_files.update({str(p.relative_to(root)):sha256_file(p) for p in paths.values()})
        for row in rows:
            records.append(review_example(row,indexed['original'][row['pair_id']],indexed['candidate_v2'][row['pair_id']],
                           data['original']['threshold'],data['candidate_v2']['threshold'],split))
    source_files.update({str(p.relative_to(root)):sha256_file(p) for p in [run/'dependency_pairs.jsonl',root/STRESS]})
    targeted=[r for r in records if (r['scenario']=='nli_context' and r['original_label']==0 and r['v2_mismatch_against_original_label'])
              or (r['split']=='stress' and r['observed_outcome_against_original_annotation'] in ['improvement','regression'])]
    return records,targeted,source_files


def summarize_reviews(records):
    counts={'unique_reviewed_rows':len(records),'findings':dict(Counter(r['finding'] for r in records)),
            'independent_nli_context_errors':sum(r['scenario']=='nli_context' and r['original_label']==0 and r['v2_mismatch_against_original_label'] for r in records),
            'validation_errors_at_least_point9':sum(r['split']=='validation' and r['v2_high_confidence_mismatch'] for r in records),
            'stress_regressions':sum(r['split']=='stress' and r['observed_outcome_against_original_annotation']=='regression' for r in records),
            'stress_improvements':sum(r['split']=='stress' and r['observed_outcome_against_original_annotation']=='improvement' for r in records),
            'human_reviewed_rows':0, 'scenarios':{},'confidence_distributions':{}}
    for scenario in sorted({r['scenario'] for r in records}):
        subset=[r for r in records if r['scenario']==scenario]
        counts['scenarios'][scenario]={'rows':len(subset),'findings':dict(Counter(r['finding'] for r in subset)),
            'mismatches':sum(r['v2_mismatch_against_original_label'] for r in subset),
            'relation_dependency_counts':dict(Counter(r['semantic_relation']+'_original_dep_'+str(r['original_label']) for r in subset)),
            'improvements_regressions':dict(Counter(r['observed_outcome_against_original_annotation'] for r in subset))}
    for split in ['validation','final','stress']:
        counts['confidence_distributions'][split]={}
        for method in ['original','candidate_v2']:
            subset=[r for r in records if r['split']==split]
            bins=[0]*10;error_bins=[0]*10;values=[]
            for r in subset:
                p=r['predictions'][method];value=p['predicted_label_probability'];values.append(value)
                index=min(9,int(value*10));bins[index]+=1
                if p['prediction'] != r['original_label']:error_bins[index]+=1
            counts['confidence_distributions'][split][method]={'origin':'captured binary softmax for threshold-selected label; not calibrated',
                'bin_edges':[i/10 for i in range(11)],'all_counts':bins,'original_annotation_mismatch_counts':error_bins,
                'min':min(values),'median':statistics.median(values),'max':max(values),'mean':statistics.mean(values),
                'annotation_limits_apply':True}
    return counts


def transformation_review(root):
    root=Path(root)
    ledger=json.loads((root/AUDIT/'paired_links.json').read_text(encoding='utf-8'))
    predictions={m:{p['pair_id']:p for p in json.loads((root/V2/'comparisons'/f'stress_{m}.json').read_text())['predictions']}
                 for m in ['original','candidate_v2']}
    reviewed=[]
    for link in ledger['pairs']+ledger['excluded_proposed_renamings']:
        a,b=link['members'];excluded='reason' in link
        controlled=link['kind']=='renaming' and not excluded or link['kind']=='role_change' and link['controlled']
        reviewed.append({'transformation_id':link['comparison_id'],'original_metadata':link,
            'review_status':'REJECTED_AS_ALIAS_ONLY' if excluded else 'CONTROLLED_TEXT_VERIFIED' if controlled else 'CONFOUNDED_ROLE_COUNTERPART',
            'factor_issue':link.get('reason') or ('Source facts and task change together; retain as counterpart only' if not controlled else 'Declared text factor verified'),
            'members':[{'case_id':key,'input':{f:predictions['original'][key][f] for f in ['premise','hypothesis']},
                        'original_label':predictions['original'][key]['label'],
                        'predictions':{m:{f:predictions[m][key][f] for f in ['prediction','probability']} for m in predictions},
                        'thresholds':{'original':.2,'candidate_v2':.2}} for key in [a,b]],
            'proposed_action':'exclude from controlled transformation scores' if not controlled else 'retain existing verified association',
            'human_reviewed':False})
    v2_links=json.loads((root/V2/'paired_links.json').read_text())['pairs']
    return {'records':reviewed,'counts':dict(Counter(r['review_status'] for r in reviewed)),
            'v2_rule_change_caveat':{'pairs':sum(l['kind']=='role_change' for l in v2_links),
                'finding':'V2 changes both support-list members, not one member with fixed cofactor; source facts are fixed. Valid rule-set-factor comparisons, not isolated single-support substitutions.',
                'v3_proposal':'one support substituted, cofactor fixed; complete factor validation'}}


def ambiguity_queue():
    return [
        {'id':'queue-no-target','premise':'Oak reports 6.','hypothesis':'Oak reports 6.','label':None,
         'reason':'NLI entailment supplies no target artifact/support contract'},
        {'id':'queue-required-not-exhaustive','premise':'Oak is signed.','hypothesis':'Release requires Pine and Elm.','label':None,
         'reason':'Other signatories may also be required; no negative dependency conclusion'},
        {'id':'queue-alternative-evidence','premise':'Oak and Pine each confirm the result.','hypothesis':'The result is verified.','label':None,
         'reason':'Declared snapshot could bind both sources or only one; sufficient alternative evidence is not declared membership'},
        {'id':'queue-zero-effect','premise':'Oak changes, output stays zero.','hypothesis':'Q is unchanged.','label':None,
         'reason':'Zero sensitivity alone does not distinguish declared support from independence'},
        {'id':'queue-full-hypothesis','premise':'Q reads only Pine and Elm. Oak reads 6.','hypothesis':'Q reads only Pine and Elm. Oak reads 6.','label':None,
         'reason':'Oak is independent of Q, but certifying the whole repeated hypothesis has a different support obligation'},
    ]


def prepare_bundle(root, out):
    root,out=Path(root),Path(out)
    if (out/'review.json').exists() or (out/'proposal/protocol.json').exists():
        raise FileExistsError('Existing review/proposal is immutable; use a new directory')
    records,targeted,sources=collect_reviews(root)
    summary=summarize_reviews(records)
    write_exclusive(out/'review.json',{'schema_version':1,'status':'RETROSPECTIVE_AUTOMATED_PROVISIONAL_REVIEW',
        'contract_version':CONTRACT_VERSION,'definition':DEFINITION,'source_files_sha256':sources,
        'summary':summary,'records':records,'targeted_review_ids':[r['review_id'] for r in targeted]})
    write_exclusive(out/'corrections.json',{'schema_version':1,'policy':'SIDECAR_ONLY; proposals never rewrite labels or historical outputs',
        'original_review_sha256':sha256_file(out/'review.json'), 'proposals':[{'review_id':r['review_id'],
        'original_row_sha256':r['original_row_sha256'],'original_label':r['original_label'],
        'proposed':r['proposed_correction'],'status':'PENDING_HUMAN_REVIEW','human_reviewed':False}
        for r in records if r['proposed_correction']]})
    write_exclusive(out/'historical_transformations.json',transformation_review(root))
    write_exclusive(out/'ambiguity_queue.json',{'policy':'ABSTAIN/EXCLUDE until explicit target and exhaustive support supplied; no forced binary labels',
        'status':'PENDING_HUMAN_REVIEW','examples':ambiguity_queue(),
        'historical_review_ids':[r['review_id'] for r in records if not r['text_sufficient_without_scope_assumptions']]})
    contrast,contrast_links=generate_collection()
    candidates,links,assignments=generate_v3_candidates()
    historical=read_jsonl(root/V2/'dependency_pairs.jsonl')+read_jsonl(root/STRESS)
    candidate_audit=audit_candidates(candidates,links,historical)
    write_exclusive(out/'contrasts.json',{'version':'mentor-contrasts-v1','annotations':'SYNTHETIC_PROVISIONAL','human_reviewed':False,
        'rows':contrast,'transformations':contrast_links,'audit':audit_candidates(contrast,contrast_links,historical)})
    proposal=out/'proposal';proposal.mkdir()
    for split in ['train','validation','future_final']:
        path=proposal/(split+'.jsonl')
        with path.open('x',encoding='utf-8') as stream:
            stream.write(''.join(canonical_json(r)+'\n' for r in candidates if r['split']==split))
    write_exclusive(proposal/'transformations.json',{'version':'candidate-v3-data-v1','links':links})
    write_exclusive(proposal/'leakage_audit.json',candidate_audit)
    v2protocol=json.loads((root/V2/'protocol.json').read_text())
    protocol={'schema_version':1,'version':'candidate-v3-plan-v1','status':'FROZEN_PROPOSAL_NOT_EXECUTED_PENDING_MENTOR',
        'frozen_utc':datetime.now(timezone.utc).isoformat(),'warning':'PRELIMINARY / EXPERIMENTAL; generated annotations provisional',
        'execution':{'allow_training':False,'allow_inference':False,'allow_downloads':False,'allow_space_requests':False},
        'dependency_contract':{'version':CONTRACT_VERSION,'definition':DEFINITION,'target':'Q only; observation check is separate'},
        'principal_intervention':'training distribution: explicit target-scoped NLI/support contrasts and declared-support versus sensitivity counterexamples',
        'controls':{'starting_model':v2protocol['model'],'training':v2protocol['training'],
            'training_rows':256,'loss':'unchanged binary cross entropy','initialization':'same pinned base, fresh binary head, same seed 42; never resume v2',
            'input_interface':'unchanged premise/hypothesis pair; new text makes target scope explicit',
            'optimizer_steps_budget':256,'budget_seconds':1800,'base_checkpoint_comparator':'frozen candidate-v2 weights and threshold .2',
            'individual_distribution_facets_not_isolated':True},
        'selection':{'checkpoint':'validation macro F1 each epoch; earliest best on ties, unchanged',
            'threshold':'validation positive F1, recall, then smallest preset candidate, unchanged',
            'prohibited':'historical stress/final and future-final scores cannot select checkpoint, threshold or hyperparameters'},
        'group_assignments':assignments,'dataset_version':'candidate-v3-data-v1',
        'dataset_file_sha256':{split:sha256_file(proposal/(split+'.jsonl')) for split in ['train','validation','future_final']},
        'transformation_file_sha256':sha256_file(proposal/'transformations.json'),
        'leakage_audit_file_sha256':sha256_file(proposal/'leakage_audit.json'),
        'historical_benchmarks':{'stress_384':'HISTORICAL/DEVELOPMENT','inspected_final_96':'HISTORICAL/DEVELOPMENT'},
        'evaluation':{'tracks':'scenario interpolation vs families held out from train vs held out from train AND validation; wording/template overlap explicit',
            'report_by':['all six NLI/dependency categories','scenario','evaluation track','alias/position','value_changed versus unchanged read supports'],
            'metrics':['accuracy','macro F1','class supports','confusion matrix','missed dependency count and rate (FN / positive support)',
                'false dependency alarm count and rate (FP / negative support)','worst scenario/category macro F1 and recall',
                'each invariant transformation equal-label count AND both-correct count','support-rule pair both-correct count',
                'unique pair coverage, shared members and overlap','captured raw probabilities/logits; high-score mismatches are descriptive, not calibration'],
            'no_future_final_predictions_in_this_phase':True},
        'acceptance_criteria':{'human_gate':'mentor approves target/scope contract, six-category contrasts and future-final labels via new approval sidecars; unknown queue excluded',
            'tokenization_gate':'offline pinned-tokenizer length/truncation audit required before execution; retain max_length 128 or amend proposal before any training',
            'primary':'at least 20 percentage-point reduction in future-final entailment/independent false alarm rate versus frozen v2 on SAME rows',
            'overall_guard':'future-final macro F1 no more than .02 below frozen v2; missed-dependency rate no more than .05 worse',
            'worst_group_guard':'each scenario binary macro F1 >= .70 and each six-category recall >= .70; report all failures even if primary passes',
            'metric_definitions':'binary macro F1 averages both classes; six contrast categories each have one true binary class, so use category recall, not single-category binary macro F1 (perfect upper bound .50)',
            'transformation_guards':'renaming equal-label rate >= .95, and controlled support-rule both-correct rate >= .80; report both-correct for invariants',
            'stability':'declared-support positive recall >= .90 on unchanged-value read scenarios',
            'interpretation':'joint criteria on a small grouped synthetic set, not statistical general accuracy or certificate safety'},
        'future_execution_prerequisites':['manual authorization for one bounded job','mentor review sidecars linked by immutable hashes',
            'tokenization/truncation audit','environment/base weight/initial head and original-v2 hash verification','full offline regression gates'],
        'authority':'advisory only; never create trusted support edges or bypass deterministic verification'}
    write_exclusive(proposal/'protocol.json',protocol)
    write_exclusive(proposal/'freeze_manifest.json',{'protocol_sha256':sha256_file(proposal/'protocol.json'),
        'file_hashes':{p.name:sha256_file(p) for p in proposal.iterdir() if p.is_file()},'status':'PREPARATION_ONLY_NO_EXECUTION'})
    make_review_sheet(out,contrast,targeted)
    render_report(out,summary,records,protocol)
    return summary


def make_review_sheet(out, contrast, targeted):
    for filename,rows in [('mentor_contrasts.csv',contrast),('mentor_errors.csv',targeted)]:
        with (out/filename).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['review_id','premise','hypothesis','original_dependency_label','NLI_interpretation',
                             'declared_rule','rationale','original_probability','v2_probability','mentor_name','review_date','decision','comments'])
            for r in rows:
                retrospective='review_id' in r
                writer.writerow([r.get('review_id',r.get('pair_id')),r['input']['premise'] if retrospective else r['premise'],
                    r['input']['hypothesis'] if retrospective else r['hypothesis'],r.get('original_label',r.get('label')),
                    r.get('semantic_relation',r.get('nli_label')),r.get('declared_rule',r.get('derivation',{}).get('rule')),
                    r.get('explanation',r.get('derivation',{}).get('label_basis')),
                    r['predictions']['original']['probability'] if retrospective else '',
                    r['predictions']['candidate_v2']['probability'] if retrospective else '', '', '', '', ''])
    content='''# Mentor review sheet

PRELIMINARY / EXPERIMENTAL. All annotations and reviews are automated/provisional.
No person has reviewed these examples. Keep original artifacts unchanged; record
decisions in new sidecars referencing row/file hashes.

1. Confirm target Q means its declared support closure, not the whole checked hypothesis.
2. Confirm zero sensitivity, redundant OR evidence and cancellation still label a declared read support 1.
3. Check all six NLI/dependency combinations: dependency is Q's support; NLI concerns the separate observation check.
4. Check one-factor renaming, fixed-cofactor support substitution, paraphrase, fact-order and irrelevant-context changes.
5. Review the 16 scope-dependent errors and 49 underdetermined stress regressions. Do not turn unknown labels into 0/1.
6. Review the 64 contrasts and 128 future-final rows before execution. Approval does not make unreviewed generated training rows human-reviewed.
7. Approve or amend the single distribution intervention and frozen acceptance rules before ANY model execution.

mentor_contrasts.csv has 64 exact examples; mentor_errors.csv has 107 distinct
priority rows (16 nli_context errors, 49 regressions, 42 improvements). Eight
high-confidence validation errors are a SUBSET of those 16, not additional rows.
Full 544-row retrospective review: review.json. Ambiguities: ambiguity_queue.json.
Current status: PENDING MENTOR. Leave reviewer/date fields blank until a real person reviews.
'''
    with (out/'MENTOR_REVIEW.md').open('x',encoding='utf-8') as stream:stream.write(content)


def render_report(out, summary, records, protocol):
    esc=lambda value:html.escape(json.dumps(value,indent=2,ensure_ascii=False))
    lines=["<!doctype html><html lang='en'><meta charset='utf-8'><title>Annotation review and v3 proposal</title><style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:20px}table{border-collapse:collapse}td,th{border:1px solid #999;padding:7px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:12px 0}</style>",
        '<h1>PRELIMINARY / EXPERIMENTAL</h1><p>Retrospective saved-response analysis, provisional synthetic annotations, and a future experiment proposal. NO training, inference, downloads or Space calls in this phase. No human review yet.</p>',
        '<p><a href="MENTOR_REVIEW.md">Mentor sheet</a> | <a href="mentor_contrasts.csv">64 contrast examples</a> | <a href="mentor_errors.csv">107 priority examples</a> | <a href="proposal/protocol.json">Frozen proposal</a> | <a href="proposal/leakage_audit.json">Overlap/coverage audit</a></p>',
        '<h2>Dependency target</h2><p>'+html.escape(DEFINITION)+'</p><p>The implemented core snapshots declared support versions, regardless of unchanged values, redundancy or cancellation. V1/v2 largely make support membership and sensitivity coincide. Their labels do not establish general sensitivity-to-support equivalence. nli_context needs an explicit Q-only scope; certifying the whole hypothesis is a different question.</p>',
        '<h2>Review counts</h2><pre>'+esc({k:v for k,v in summary.items() if k not in ['scenarios','confidence_distributions']})+'</pre>',
        '<p>All 49 stress regressions are underdetermined NLI-trap negatives. No binary label inversion is established. Of 42 measured improvements, 22 arithmetic examples have an exhaustive task rule; 20 approval negatives rely on an unstated exhaustivity assumption. Two v2 arithmetic errors remain under explicit provisional rules. Sixteen nli_context errors are conditional on the saved Q-only scope. Repeated text or score differences do not demonstrate causal memorization.</p>',
        '<h2>Counts by scenario / semantic relation / dependency annotation</h2><pre>'+esc(summary['scenarios'])+'</pre>',
        '<h2>Saved confidence distributions</h2><p>Full-precision binary softmax for the threshold-selected label. Scores are advisory, not calibrated. Mismatch bins compare original provisional labels, including unresolved annotations.</p>']
    for split,methods in summary['confidence_distributions'].items():
        lines.append('<h3>'+split+'</h3><table><tr><th>Model</th><th>Bin [0,.1)..[.9,1]</th><th>All rows</th><th>Original-label mismatches</th></tr>')
        for method,stats in methods.items():
            for i,(n,e) in enumerate(zip(stats['all_counts'],stats['original_annotation_mismatch_counts'])):
                lines.append(f'<tr><td>{method}</td><td>{i/10:.1f}–{(i+1)/10:.1f}</td><td>{n}</td><td>{e}</td></tr>')
        lines.append('</table>')
    lines.append('<h2>Exact priority examples</h2><p>Inputs, saved labels, both predictions/probabilities/thresholds, derivations and sidecar proposals follow. Complete records remain in review.json.</p>')
    for r in records:
        if r['v2_mismatch_against_original_label'] or r['observed_outcome_against_original_annotation']=='improvement':
            lines.append('<details><summary>'+html.escape(r['review_id']+' — '+r['finding'])+'</summary><pre>'+esc(r)+'</pre></details>')
    lines.extend(['<h2>Annotations and transformations</h2><p><a href="corrections.json">Immutable proposed corrections</a> | <a href="historical_transformations.json">Rejected/confounded historical pair constructions</a> | <a href="ambiguity_queue.json">Unknowns excluded pending policy/context</a> | <a href="contrasts.json">Exact synthetic contrast inputs, rationales and transformations</a></p>',
        '<h2>Future candidate v3 — NOT EXECUTED</h2><pre>'+esc(protocol)+'</pre>',
        '<p>The principal change is training distribution. Base, binary loss/head, training size, seed, two-epoch/256-step budget and selection policy remain fixed. Individual distribution facets are not isolated. Structural/template overlap is explicitly allowed for interpolation, separately from held-out families. Historical 384-row stress and observed 96-row final are DEVELOPMENT evidence.</p>',
        '<p>Contradictory observations can coexist with declared Q dependency: no certificate for the false checked statement is authorized. All model advice remains advisory.</p></html>'])
    with (out/'report.html').open('x',encoding='utf-8') as stream:stream.write(''.join(lines))
