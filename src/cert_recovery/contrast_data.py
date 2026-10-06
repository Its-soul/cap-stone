"""Synthetic/provisional target-scoped contrasts; no model imports or execution."""
from collections import Counter
import copy
import re

from .crypto import sha256
from .data_pipeline import normalize_text
from .dependency_contract import CONTRACT_VERSION, FORMULAS, dependency_label, contract_witness
from .paired_metrics import rename_text, align_predictions

ALIASES = ['Oak', 'Pine', 'Elm', 'Fir']
RELATIONS = ['entailment', 'contradiction', 'neutral']
TEMPLATES = [
    "Q in {world} reads ONLY {a} and {b}; rule: {formula}; candidate: {candidate}. Observation check is separate from Q support.",
    "Record {world}: Q declares exactly {a}, {b} as read supports; computes {formula}; source queried: {candidate}. Check concerns observations only.",
    "At {world}, Q's exhaustive read support is {a} and {b}; operation: {formula}; candidate {candidate}. Separate observation check follows.",
]
FACTS = '{name} reads {value}.'


def nli_label(fields):
    check = fields['check']
    if check['property'] == 'warranty':
        return 'neutral'  # Neither active nor inactive warranty is asserted in the facts.
    if check['property'] != 'reading' or check['source'] not in fields['state']:
        raise ValueError('Unsupported observation check')
    if check['source'] not in fields['fact_order']:
        return 'neutral'  # Hidden generator state is not evidence in the premise.
    return 'entailment' if check['value'] == fields['state'][check['source']] else 'contradiction'


def render(fields):
    a,b = fields['supports']
    context = TEMPLATES[fields['template']].format(world=fields['world'], a=a, b=b,
        formula=FORMULAS[fields['rule_kind']], candidate=fields['candidate'])
    facts = ' '.join(FACTS.format(name=s,value=fields['state'][s]) for s in fields['fact_order'])
    check = fields['check']
    if check['property'] == 'reading':
        statement = (f"{check['source']} reads {check['value']}." if fields['surface_style'] == 'literal'
                     else f"The reported reading of {check['source']} equals {check['value']}.")
    else:
        statement = (f"{check['source']} has an active warranty." if fields['surface_style'] == 'literal'
                     else f"A current warranty covers {check['source']}.")
    premise = context + ' Facts: ' + facts
    if fields['extra_context']:
        premise += ' ' + fields['extra_context']
    hypothesis = context + ' Check: ' + statement
    return premise, hypothesis, {'premise':facts, 'hypothesis':statement,
                                'scope':'observation facts -> checked statement; context repeated in full model input'}


def make_row(pair_id, fields, group, split, family, evaluation_track, parent=None):
    graph = {'Q': fields['supports'], **{name: [] for name in fields['state']}}
    label = dependency_label(fields['candidate'], 'Q', graph)
    premise,hypothesis,projection = render(fields)
    witness = contract_witness(fields['rule_kind'],fields['state'],fields['supports'],fields['candidate'])
    return {'pair_id':pair_id, 'world_id':group, 'group_id':group, 'split':split,
            'template_family':family, 'scenario':fields['rule_kind'], 'evaluation_track':evaluation_track,
            'premise':premise, 'hypothesis':hypothesis, 'label':label, 'nli_label':nli_label(fields),
            'contrast_category':nli_label(fields)+'_dependency_'+str(label), 'nli_projection':projection,
            'parent_pair_id':parent, 'annotation_status':'SYNTHETIC_PROVISIONAL_PENDING_HUMAN_REVIEW',
            'human_reviewed':False, 'source':'candidate-v3 preparation; no model outputs',
            'contract_version':CONTRACT_VERSION, 'fields':copy.deepcopy(fields),
            'derivation':{'target':'Q only; not the complete hypothesis', 'declared_support_graph':graph,
                'complete_nodes':sorted(graph), 'rule':FORMULAS[fields['rule_kind']], 'intervention':witness,
                'label_basis':'declared ancestry membership, irrespective of observed value change'}}


def _link(parent, child, kind, factor, alias_map=None):
    return {'transformation_id':parent['pair_id']+'__'+kind+'__'+child['pair_id'],
            'parent_pair_id':parent['pair_id'], 'child_pair_id':child['pair_id'], 'kind':kind,
            'declared_factor':factor, 'alias_map':alias_map,
            'expected_dependency_labels':[parent['label'],child['label']],
            'expected_nli_labels':[parent['nli_label'],child['nli_label']]}


def _rename_fields(fields, mapping):
    f = copy.deepcopy(fields)
    f['state'] = {mapping[k]:v for k,v in f['state'].items()}
    for key in ['supports','fact_order']:
        f[key] = [mapping[k] for k in f[key]]
    f['candidate'] = mapping[f['candidate']]
    f['check']['source'] = mapping[f['check']['source']]
    return f


def generate_group(group, split, scenario, template, candidate_index, position, evaluation_track, ordinal):
    aliases = ALIASES
    candidate = aliases[candidate_index]
    cofactor, substitute, extra = [a for a in aliases if a != candidate]
    state = {a: (1 if scenario in {'redundant_or','all_signoff','both_valid'} else 2+(i+ordinal)%5) for i,a in enumerate(aliases)}
    # Distinct unknown warranties are not silently treated as false observations.
    facts = [candidate,cofactor] if ordinal%2 == 0 else [cofactor,candidate]
    mapping = {aliases[i]:aliases[(i+1)%4] for i in range(4)}
    family = f'{scenario}-t{template}'
    rows, links, base, renamed = [], [], {}, {}
    for relation in RELATIONS:
        for label in (1,0):
            source = candidate if label else substitute
            supports = [source,cofactor] if position == 0 else [cofactor,source]
            check = {'source':candidate,'property':'warranty' if relation=='neutral' else 'reading',
                     'value':None if relation=='neutral' else state[candidate]+int(relation=='contradiction')}
            fields = {'world':group, 'candidate':candidate, 'supports':supports, 'state':state,
                      'fact_order':facts, 'rule_kind':scenario, 'template':template,
                      'check':check,'surface_style':'literal','extra_context':''}
            key = (relation,label)
            parent = f'{group}-{relation}-y1-base' if label == 0 else None
            row = make_row(f'{group}-{relation}-y{label}-base',fields,group,split,family,evaluation_track,parent)
            rows.append(row); base[key] = row
            renamed_row = make_row(f'{group}-{relation}-y{label}-rename',_rename_fields(fields,mapping),group,split,family,evaluation_track,row['pair_id'])
            rows.append(renamed_row); renamed[key] = renamed_row
            links.append(_link(row,renamed_row,'renaming','identifier_bijection',mapping))
        links.append(_link(base[(relation,1)],base[(relation,0)],'support_rule','supports'))
        links.append(_link(renamed[(relation,1)],renamed[(relation,0)],'support_rule','supports'))
    # Two paired invariant views keep each alias/label equally represented.
    kinds = ['lexical_similarity','facts_reorder','irrelevant_context']
    for view_index, source in enumerate((base,renamed)):
        kind = kinds[(ordinal+view_index)%3]
        relation = RELATIONS[(ordinal+view_index)%3]
        variants = {}
        for label in (1,0):
            parent = source[(relation,label)]
            f = copy.deepcopy(parent['fields'])
            if kind == 'lexical_similarity': f['surface_style'] = 'paraphrase'
            elif kind == 'facts_reorder': f['fact_order'] = list(reversed(f['fact_order']))
            else: f['extra_context'] = 'Unrelated room has a blue door.'
            row = make_row(f'{group}-{relation}-y{label}-{kind}-{view_index}',f,group,split,family,evaluation_track,parent['pair_id'])
            rows.append(row); variants[label] = row
            factor = {'lexical_similarity':'surface_style','facts_reorder':'fact_order','irrelevant_context':'extra_context'}[kind]
            links.append(_link(parent,row,kind,factor))
        links.append(_link(variants[1],variants[0],'support_rule','supports'))
    return rows, links


def generate_collection():
    rows, links = [], []
    for i, scenario in enumerate(['sum','zero_read','cancellation','redundant_or']):
        a,b = generate_group(f'contrast_w{i:04d}','review',scenario,0,i,i%2,'mentor_contrast',i)
        rows.extend(a); links.extend(b)
    validate_transformations(rows, links)
    return rows, links


def generate_v3_candidates():
    rows, links, assignments = [], [], {}
    specs = [('train',s,0,'training') for s in ['sum','zero_read','cancellation','redundant_or'] for _ in range(4)]
    specs += [('validation',s,0 if i<2 else 1,'scenario_interpolation' if i<2 else 'family_held_out_from_training')
              for i,s in enumerate(['sum','zero_read','weighted_sum','all_signoff'])]
    specs += [('future_final',s,0 if i<4 else 1 if i<6 else 2,
               'scenario_interpolation' if i<4 else 'family_held_out_from_training' if i<6 else 'family_held_out_from_train_and_validation')
              for i,s in enumerate(['sum','zero_read','cancellation','redundant_or','weighted_sum','all_signoff','both_valid','max_read'])]
    for i,(split,scenario,template,track) in enumerate(specs):
        group = f'v3_w{i:04d}'
        assignments[group] = split
        a,b = generate_group(group,split,scenario,template,i%4,i%2,track,i)
        rows.extend(a); links.extend(b)
    validate_transformations(rows, links)
    return rows, links, assignments


def validate_row(row):
    f = row['fields']
    graph = {'Q':f['supports'],**{name:[] for name in f['state']}}
    if dependency_label(f['candidate'],'Q',graph) != row['label'] or nli_label(f) != row['nli_label']:
        raise ValueError('Dependency/NLI annotation does not match explicit contract')
    p,h,nli = render(f)
    if (p,h,nli) != (row['premise'],row['hypothesis'],row['nli_projection']):
        raise ValueError('Input text/projection differs from declared factors')
    if row['derivation']['declared_support_graph'] != graph or row['contrast_category'] != row['nli_label']+'_dependency_'+str(row['label']):
        raise ValueError('Derivation/category mismatch')
    if (row['scenario'] != f['rule_kind'] or row['world_id'] != row['group_id'] or f['world'] != row['group_id']
            or row['template_family'] != f"{f['rule_kind']}-t{f['template']}"
            or row['contract_version'] != CONTRACT_VERSION
            or row['derivation']['complete_nodes'] != sorted(graph)
            or row['derivation']['rule'] != FORMULAS[f['rule_kind']]):
        raise ValueError('Contract/group/scenario metadata mismatch')
    w = row['derivation']['intervention']
    after_state = {**f['state'], **w['changes']}
    from .dependency_contract import computed_value
    before = computed_value(f['rule_kind'], f['state'], f['supports'])
    after = computed_value(f['rule_kind'], after_state, f['supports'])
    if (w['dependency_label'], w['before'], w['after'], w['value_changed']) != (row['label'], before, after, before != after):
        raise ValueError('Intervention witness contradicts explicit rule')
    if row['human_reviewed'] is not False or row['annotation_status'] != 'SYNTHETIC_PROVISIONAL_PENDING_HUMAN_REVIEW':
        raise ValueError('Generated annotations must not claim human review')


def validate_transformations(rows, links):
    indexed = {r['pair_id']:r for r in rows}
    if len(indexed) != len(rows): raise ValueError('Duplicate input ID')
    for row in rows: validate_row(row)
    seen, edges = set(), set()
    for link in links:
        name = link['transformation_id']
        if name in seen: raise ValueError('Duplicate transformation ID')
        seen.add(name)
        a,b = [indexed.get(link[k]) for k in ['parent_pair_id','child_pair_id']]
        if a is None or b is None or a['pair_id'] == b['pair_id']: raise ValueError('Missing/duplicated pair member')
        edge = (a['pair_id'],b['pair_id'],link['kind'])
        if edge in edges: raise ValueError('Duplicate transformation members')
        edges.add(edge)
        if a['group_id'] != b['group_id'] or a['split'] != b['split']: raise ValueError('Related variants must share group/split')
        if [a['label'],b['label']] != link['expected_dependency_labels'] or [a['nli_label'],b['nli_label']] != link['expected_nli_labels']:
            raise ValueError('Mismatched expected relationships')
        kind = link['kind']; expected = copy.deepcopy(a['fields'])
        if kind == 'renaming':
            mapping = link['alias_map']
            rename_text(a['premise'],mapping)  # Existing bijection checks.
            expected = _rename_fields(expected,mapping)
            if any(rename_text(a[f],mapping) != b[f] for f in ['premise','hypothesis']):
                raise ValueError('Renaming also changed non-alias text')
        elif kind == 'support_rule':
            if a['label'] != 1 or b['label'] != 0 or len(set(a['fields']['supports']) & set(b['fields']['supports'])) != 1:
                raise ValueError('Role change must substitute one support, keeping cofactor')
            expected['supports'] = b['fields']['supports']
        elif kind == 'lexical_similarity': expected['surface_style'] = 'paraphrase'
        elif kind == 'facts_reorder': expected['fact_order'] = list(reversed(expected['fact_order']))
        elif kind == 'irrelevant_context': expected['extra_context'] = 'Unrelated room has a blue door.'
        else: raise ValueError('Unknown transformation')
        factor = {'renaming':'identifier_bijection','support_rule':'supports','lexical_similarity':'surface_style',
                  'facts_reorder':'fact_order','irrelevant_context':'extra_context'}[kind]
        if link['declared_factor'] != factor or expected != b['fields'] or a['fields'] == b['fields']:
            raise ValueError('Undeclared factor change or no-op transformation')
        if a['nli_label'] != b['nli_label'] or (kind != 'support_rule' and a['label'] != b['label']):
            raise ValueError('Unexpected NLI/dependency label change')
    return indexed


def audit_candidates(rows, links, historical_rows=()):
    validate_transformations(rows,links)
    splits = sorted({r['split'] for r in rows})
    train_families = {r['scenario'] for r in rows if r['split'] == 'train'}
    validation_families = {r['scenario'] for r in rows if r['split'] == 'validation'}
    for row in rows:
        if row['split'] == 'review':
            expected_track = 'mentor_contrast'
        elif row['split'] == 'train':
            expected_track = 'training'
        elif row['scenario'] in train_families:
            expected_track = 'scenario_interpolation'
        elif row['split'] == 'validation' or row['scenario'] in validation_families:
            expected_track = 'family_held_out_from_training'
        else:
            expected_track = 'family_held_out_from_train_and_validation'
        if row['evaluation_track'] != expected_track:
            raise ValueError('Evaluation track contradicts observed family overlap')
    exact = lambda r:(r['premise'],r['hypothesis'])
    normalized = lambda r:tuple(normalize_text(s).casefold() for s in exact(r))
    def structural(r):
        return tuple(re.sub(r'\b\d+\b','<number>',re.sub(r'\b(?:contrast_w|v3_w)\d+\b','<world>',
            re.sub(r'\b(?:Oak|Pine|Elm|Fir)\b','<alias>',s))).casefold() for s in exact(r))
    groups = {}
    intersections = []
    for split in splits:
        subset = [r for r in rows if r['split']==split]
        for project in [exact,normalized]:
            if len({project(r) for r in subset}) != len(subset): raise ValueError('Duplicate input pairs')
        aliases = {alias:[sum(r['label']==label and r['fields']['candidate']==alias for r in subset) for label in (0,1)] for alias in ALIASES}
        positions = {alias:[sum(r['label']==1 and r['fields']['candidate']==alias and r['fields']['supports'].index(alias)==pos for r in subset) for pos in (0,1)] for alias in ALIASES}
        if len({tuple(v) for v in aliases.values()}) != 1 or any(a!=b for a,b in aliases.values()):
            raise ValueError('Alias/label counterbalancing failed')
        if any(a!=b for a,b in positions.values()): raise ValueError('Source-position counterbalancing failed')
        groups[split] = {'rows':len(subset),'groups':sorted({r['group_id'] for r in subset}),
            'categories':dict(Counter(r['contrast_category'] for r in subset)), 'aliases_by_label':aliases,
            'positive_support_positions':positions,'scenarios':dict(Counter(r['scenario'] for r in subset)),
            'evaluation_tracks':dict(Counter(r['evaluation_track'] for r in subset)),
            'template_families':sorted({r['template_family'] for r in subset}), 'canonical_sha256':sha256(subset)}
    for i,a in enumerate(splits):
        left = [r for r in rows if r['split']==a]
        for b in splits[i+1:]:
            right = [r for r in rows if r['split']==b]
            if {r['group_id'] for r in left} & {r['group_id'] for r in right}: raise ValueError('Group leakage')
            if {normalized(r) for r in left} & {normalized(r) for r in right}: raise ValueError('Normalized leakage')
            shared = {structural(r) for r in left} & {structural(r) for r in right}
            left_families={r['template_family'] for r in left};right_families={r['template_family'] for r in right}
            # Structural overlap is intentional only for declared interpolation families.
            unexplained = {structural(r) for r in left if r['template_family'] not in right_families} & {structural(r) for r in right}
            unexplained |= {structural(r) for r in right if r['template_family'] not in left_families} & {structural(r) for r in left}
            if unexplained: raise ValueError('Undeclared structural/template leakage')
            intersections.append({'splits':[a,b],'shared_structural_pairs':len(shared),
                'shared_template_families':sorted(left_families & right_families),
                'interpretation':'Declared interpolation overlap; never claimed fully template-disjoint'})
    if {normalized(r) for r in rows} & {normalized(r) for r in historical_rows}: raise ValueError('Historical inputs reused')
    return {'status':'PASS','split_audits':groups,'overlap_audit':intersections,
            'exact_duplicates':0,'normalized_duplicates':0,'historical_input_overlap':0,
            'transformation_counts':dict(Counter(t['kind'] for t in links)),
            'scope':'Specified text/template projections, not proof of independence or annotation validity'}


def transformation_metrics(rows, predictions, links):
    indexed,aligned = align_predictions(rows,predictions)
    validate_transformations(rows,links)
    result = {}
    for kind in sorted({l['kind'] for l in links}):
        selected = [l for l in links if l['kind']==kind]
        correct = same = 0; members = Counter()
        for l in selected:
            a,b = l['parent_pair_id'],l['child_pair_id'];members.update([a,b])
            same += aligned[a]['prediction']==aligned[b]['prediction']
            correct += all(aligned[k]['prediction']==indexed[k]['label'] for k in [a,b])
        result[kind]={'numerator':correct if kind=='support_rule' else same,'denominator':len(selected),
                      'both_correct_numerator':correct,'unique_rows':len(members),'repeated_members':sum(n>1 for n in members.values()),
                      'definition':'both expected opposite labels correct' if kind=='support_rule' else 'equal predictions; report both-correct separately'}
    return result
