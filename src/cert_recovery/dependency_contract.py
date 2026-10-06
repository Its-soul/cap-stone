"""Offline annotation contract, with no prediction or dependency-edge authority."""
from .engine import CertificateSystem
from .models import ArtifactKind as K, VerificationResult

CONTRACT_VERSION = 'declared-support-closure-v1'
DEFINITION = ('For a distinct candidate source S and named target artifact T, label 1 means '
              'S belongs to the explicitly declared transitive support ancestry of T; '
              'label 0 requires an exhaustive support contract proving it absent. '
              'Unknown target or incomplete ancestry yields no binary annotation. '
              'This does not predict value sensitivity, minimal causal necessity, NLI truth or similarity.')


def dependency_label(candidate, target, graph, complete_nodes=None):
    if not candidate or not target or candidate == target or target not in graph:
        raise ValueError('Distinct source and known target required')
    complete = set(graph) if complete_nodes is None else set(complete_nodes)
    visiting, visited, closure = set(), set(), set()
    incomplete = False
    def visit(node):
        nonlocal incomplete
        if node in visiting:
            raise ValueError('Cyclic declared support contract')
        if node in visited:
            return
        visiting.add(node)
        parents = graph.get(node, [])
        if not isinstance(parents, list) or len(set(parents)) != len(parents) or any(not isinstance(p, str) or not p for p in parents):
            raise ValueError('Supports require unique nonempty IDs')
        incomplete |= node not in complete
        for parent in parents:
            closure.add(parent)
            visit(parent)
        visiting.remove(node)
        visited.add(node)
    visit(target)
    return 1 if candidate in closure else None if incomplete else 0


FORMULAS = {'sum': 'their sum', 'zero_read': 'zero times their sum',
            'cancellation': 'each read value minus itself, summed',
            'redundant_or': 'boolean OR of both read flags', 'weighted_sum': 'twice the first plus three times the second',
            'all_signoff': 'both read signing flags true', 'both_valid': 'both read validation flags true',
            'max_read': 'maximum of both read values', 'difference': 'first read minus second read'}


def computed_value(kind, state, supports):
    if kind not in FORMULAS or len(supports) != 2:
        raise ValueError('Known two-input rule required')
    # Eager reads and explicit bindings: zero coefficients/redundancy never prune support.
    a, b = [state[name] for name in supports]
    return {'sum': lambda: a+b, 'zero_read': lambda: 0*(a+b),
            'cancellation': lambda: (a-a)+(b-b), 'redundant_or': lambda: int(bool(a) or bool(b)),
            'weighted_sum': lambda: 2*a+3*b, 'all_signoff': lambda: int(bool(a) and bool(b)),
            'both_valid': lambda: int(bool(a) and bool(b)), 'max_read': lambda: max(a,b),
            'difference': lambda: a-b}[kind]()


def contract_witness(kind, state, supports, candidate, changes=None):
    """Synthetic correctness fixture using the unchanged core; never model advice."""
    graph = {'Q': list(supports), **{name: [] for name in state}}
    label = dependency_label(candidate, 'Q', graph)
    system = CertificateSystem()
    refs = {name: system.add_dependency(name, {'value': value}) for name, value in state.items()}
    def compute(nodes):
        return computed_value(kind, {n.ref.artifact_id: n.payload['value'] for n in nodes if n.kind == K.DEPENDENCY}, supports)
    system.register_verifier('declared-function', lambda subject, ancestors:
        VerificationResult(subject.payload['value'] == compute(ancestors), 'declared function on exact support versions'))
    before = computed_value(kind, state, supports)
    q = system.add_artifact('Q', K.CLAIM, {'value': before}, tuple(refs[s] for s in supports))
    cert = system.issue_certificate('C', q, 'declared-function')
    system.register_rebuilder('Q', lambda old, current: {'value': compute(current)})
    if changes is None:
        changes = {candidate: 0 if kind in {'redundant_or', 'all_signoff', 'both_valid'} else state[candidate]+1}
    affected = set()
    for name, value in changes.items():
        impact = system.change_dependency(name, {'value': value})
        affected.update(ref.artifact_id for ref in impact.affected)
    valid_before_recovery = system.is_valid(cert)
    system.execute_recovery(system.plan_pending_recovery())
    after = system.latest('Q').payload['value']
    return {'dependency_label': label, 'value_changed': before != after, 'before': before, 'after': after,
            'changes': changes, 'affected_ids': sorted(affected),
            'old_certificate_valid_after_change': valid_before_recovery,
            'old_certificate_status_after_recovery': system.certificate_status(cert).value,
            'new_certificate_version': system.latest('C').ref.version,
            'current_certificate_valid': system.is_valid(system.latest('C').ref),
            'declared_supports': [system.artifact(ref).ref.to_dict() for ref in system.latest('Q').supports],
            'audit_events': [entry['kind'] for entry in system.audit_export()]}
