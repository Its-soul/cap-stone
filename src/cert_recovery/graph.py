from __future__ import annotations

from .models import Artifact, ArtifactKind, Ref


class DependencyGraph:
    """Edges point from support to consumer. Existing versions are never edited."""

    def __init__(self) -> None:
        self._nodes: dict[Ref, Artifact] = {}
        self._latest: dict[str, Ref] = {}
        self._children: dict[Ref, set[Ref]] = {}

    def get(self, ref: Ref) -> Artifact:
        return self._nodes[ref]

    def latest_ref(self, artifact_id: str) -> Ref:
        return self._latest[artifact_id]

    def latest(self, artifact_id: str) -> Artifact:
        return self.get(self.latest_ref(artifact_id))

    def all_nodes(self) -> tuple[Artifact, ...]:
        return tuple(self._nodes[ref] for ref in sorted(self._nodes))

    def current_nodes(self) -> tuple[Artifact, ...]:
        return tuple(self.get(self._latest[key]) for key in sorted(self._latest))

    def add(self, node: Artifact) -> None:
        if node.ref in self._nodes or len(set(node.supports)) != len(node.supports):
            raise ValueError("Duplicate artifact reference or support")
        previous = self._latest.get(node.ref.artifact_id)
        if node.ref.version != (previous.version + 1 if previous else 1):
            raise ValueError("Versions must increase by exactly one")
        if previous and self.get(previous).kind != node.kind:
            raise ValueError("An artifact ID cannot change kind")
        for support in node.supports:
            if support not in self._nodes:
                raise KeyError(f"Missing support: {support}")
            # Logical-ID cycles are forbidden even when older versions form a temporal DAG.
            if support.artifact_id == node.ref.artifact_id or any(
                item.artifact_id == node.ref.artifact_id for item in self.ancestors(support)
            ):
                raise ValueError("Logical dependency cycle")
        if not node.integrity_valid():
            raise ValueError("Invalid artifact hash")
        self._nodes[node.ref] = node
        self._latest[node.ref.artifact_id] = node.ref
        self._children.setdefault(node.ref, set())
        for support in node.supports:
            self._children[support].add(node.ref)

    def direct_dependents(self, ref: Ref) -> tuple[Ref, ...]:
        self.get(ref)
        return tuple(sorted(self._children[ref]))

    def descendants(self, ref: Ref) -> set[Ref]:
        self.get(ref)
        seen: set[Ref] = set()
        pending = list(self._children[ref])
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(self._children[current])
        return seen

    def ancestors(self, ref: Ref) -> set[Ref]:
        seen: set[Ref] = set()
        pending = list(self.get(ref).supports)
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(self.get(current).supports)
        return seen

    def affected_certificates(self, ref: Ref) -> tuple[Ref, ...]:
        return tuple(sorted(item for item in self.descendants(ref)
                            if self.get(item).kind == ArtifactKind.CERTIFICATE))

    def topological_order(self, refs: set[Ref], priority=None) -> tuple[Ref, ...]:
        if not refs.issubset(self._nodes):
            raise KeyError("Unknown node in recovery set")
        key = priority or (lambda ref: (ref.artifact_id, ref.version))
        remaining = {ref: set(self.get(ref).supports).intersection(refs) for ref in refs}
        result: list[Ref] = []
        while remaining:
            ready = sorted((ref for ref, parents in remaining.items() if not parents), key=key)
            if not ready:
                raise ValueError("Dependency cycle")
            current = ready[0]
            result.append(current)
            del remaining[current]
            for parents in remaining.values():
                parents.discard(current)
        return tuple(result)

