"""Mathematical L1/L2/L3 toroidal body for HTCE-Origin Clean Body.

L1/L2/L3 body implements deterministic Q256 layer
state transitions.  The layer body is still bounded and simulation-first: it
stores no raw language truth by itself and it never bypasses AIR, proof,
evidence, policy or trace boundaries.

Mathematical contract
---------------------
Runtime layer state is an element of a finite discrete torus:

    h_l(t) in T_N^d = (Z / N Z)^d, N = 2**256 by default in the Q256 profile.

A transition is an integer phase commit:

    h_l(t+1) = h_l(t) + Δ_l(t) mod N.

Inter-level projections are deterministic integer maps:

    π_{a→b}(v)_j = b_j + Σ_i c_{j,i}^{a,b} v_i mod N,
    c_{j,i}^{a,b} in {-1, 0, +1}.

No runtime float is used.  Projection coefficients are generated from a stable
hash seed so that the body is reproducible across runs and machines.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

from htce_origin.kernel.core import DEFAULT_TORUS_DIMENSION, FactDelta, TorusVector, active_state_digest, hash_to_phase
from htce_origin.kernel.q16 import DEFAULT_MODULUS, q_add, q_mod, q_sub, q_vector_add, q_vector_sub


class LayerError(ValueError):
    """Raised when a layer transition violates the toroidal body contract."""


class LayerName(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


@dataclass(frozen=True)
class LayerState:
    """Immutable state of one toroidal layer.

    ``vector`` is always normalized in ``(Z/NZ)^d``.  ``digest`` is computed from
    the layer name, vector, clock and metadata, and therefore changes whenever a
    mathematically visible transition is committed.
    """

    name: LayerName | str
    vector: tuple[int, ...]
    clock: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)
    modulus: int = DEFAULT_MODULUS
    digest: str = ""

    def __post_init__(self) -> None:
        layer = self.name if isinstance(self.name, LayerName) else LayerName(str(self.name))
        if self.clock < 0:
            raise LayerError("layer clock must be non-negative")
        normalized = tuple(q_mod(v, self.modulus) for v in self.vector)
        if not normalized:
            raise LayerError("layer vector must be non-empty")
        payload = {
            "clock": self.clock,
            "metadata": dict(self.metadata),
            "name": layer.value,
            "vector": normalized,
        }
        object.__setattr__(self, "name", layer)
        object.__setattr__(self, "vector", normalized)
        object.__setattr__(self, "digest", self.digest or active_state_digest(payload))

    @classmethod
    def zero(cls, name: LayerName | str, *, dimension: int = DEFAULT_TORUS_DIMENSION, modulus: int = DEFAULT_MODULUS) -> "LayerState":
        if dimension <= 0:
            raise LayerError("layer dimension must be positive")
        return cls(name=name, vector=tuple(0 for _ in range(dimension)), modulus=modulus)

    @property
    def dimension(self) -> int:
        return len(self.vector)

    def apply(self, delta: Iterable[int], *, evidence_id: str, metadata: Mapping[str, object] | None = None) -> "LayerState":
        delta_values = tuple(q_mod(v, self.modulus) for v in delta)
        if len(delta_values) != self.dimension:
            raise LayerError("layer delta dimension mismatch")
        next_metadata = dict(self.metadata)
        next_metadata.update(dict(metadata or {}))
        next_metadata["last_evidence_id"] = str(evidence_id)
        return LayerState(
            name=self.name,
            vector=q_vector_add(self.vector, delta_values, self.modulus),
            clock=self.clock + 1,
            metadata=next_metadata,
            modulus=self.modulus,
        )


@dataclass(frozen=True)
class LayerDelta:
    target_layer: LayerName | str
    delta: tuple[int, ...]
    evidence_id: str
    source_layer: LayerName | str | None = None
    reason: str = ""
    modulus: int = DEFAULT_MODULUS

    def __post_init__(self) -> None:
        target = self.target_layer if isinstance(self.target_layer, LayerName) else LayerName(str(self.target_layer))
        source = self.source_layer
        if source is not None and not isinstance(source, LayerName):
            source = LayerName(str(source))
        if not self.evidence_id:
            raise LayerError("layer delta evidence_id must be non-empty")
        object.__setattr__(self, "target_layer", target)
        object.__setattr__(self, "source_layer", source)
        object.__setattr__(self, "delta", tuple(q_mod(v, self.modulus) for v in self.delta))


class InterLevelProjection:
    """Deterministic integer projection between toroidal layers."""

    def __init__(self, *, dimension: int = DEFAULT_TORUS_DIMENSION, modulus: int = DEFAULT_MODULUS, seed: str = "HTCE-L123-v1") -> None:
        if dimension <= 0:
            raise LayerError("projection dimension must be positive")
        self.dimension = int(dimension)
        self.modulus = int(modulus)
        self.seed = str(seed)

    def project(self, vector: Iterable[int], *, source: LayerName | str, target: LayerName | str) -> tuple[int, ...]:
        values = tuple(q_mod(v, self.modulus) for v in vector)
        if len(values) != self.dimension:
            raise LayerError("projection input dimension mismatch")
        src = source if isinstance(source, LayerName) else LayerName(str(source))
        dst = target if isinstance(target, LayerName) else LayerName(str(target))
        if src == dst:
            return values
        result: list[int] = []
        for j in range(self.dimension):
            acc = self._bias(src, dst, j)
            for i, phase in enumerate(values):
                coeff = self._coeff(src, dst, j, i)
                if coeff == 1:
                    acc += phase
                elif coeff == -1:
                    acc -= phase
            result.append(q_mod(acc, self.modulus))
        return tuple(result)

    def _coeff(self, source: LayerName, target: LayerName, row: int, col: int) -> int:
        digest = hashlib.blake2b(
            f"{self.seed}|{source.value}>{target.value}|{row}|{col}".encode("utf-8"),
            digest_size=1,
            person=b"L123coef",
        ).digest()[0]
        rem = digest % 3
        if rem == 0:
            return -1
        if rem == 1:
            return 0
        return 1

    def _bias(self, source: LayerName, target: LayerName, row: int) -> int:
        digest = hashlib.blake2b(
            f"{self.seed}|bias|{source.value}>{target.value}|{row}".encode("utf-8"),
            digest_size=4,
            person=b"L123bias",
        ).digest()
        return int.from_bytes(digest, "big") % self.modulus



def _observation_attr(observation: object, name: str, default: object | None = None) -> object:
    """Read observation attribute or mapping entry without importing control."""
    if isinstance(observation, Mapping):
        return observation.get(name, default)
    return getattr(observation, name, default)


def simulated_observation_delta(
    observation: object,
    *,
    dimension: int = DEFAULT_TORUS_DIMENSION,
    modulus: int = DEFAULT_MODULUS,
) -> tuple[int, ...]:
    """Map a simulation-only observation into an L1 phase delta.

    This is the body-side map L1.observe_simulated(o_t) -> Δ_1(t). It does not
    commit L2/L3 facts and refuses observations that claim real sensor commit
    authority.
    """
    real_allowed = bool(_observation_attr(observation, "real_sensor_commit_allowed", False))
    if real_allowed:
        raise LayerError("real sensor commit is blocked in v0.1")
    modality = str(_observation_attr(observation, "modality", "simulated"))
    value = str(_observation_attr(observation, "value", "observation"))
    intensity = int(_observation_attr(observation, "intensity_bp", 0))
    reliability = int(_observation_attr(observation, "reliability_bp", 0))
    phase_obj = _observation_attr(observation, "phase", ())
    phase = tuple(q_mod(value, modulus) for value in phase_obj) if phase_obj else ()
    if not phase:
        phase = hash_to_phase(f"simulated-observation:{modality}:{value}", dimension=dimension, modulus=modulus, namespace="l1_simulated_observation_fallback")
    payload = {
        "intensity_bp": intensity,
        "modality": modality,
        "phase": phase,
        "real_sensor_commit_allowed": False,
        "reliability_bp": reliability,
        "value": value,
    }
    return hash_to_phase(payload, dimension=dimension, modulus=modulus, namespace="l1_simulated_observation")


@dataclass(frozen=True)
class BodyTransition:
    source_layer: LayerName
    target_layer: LayerName
    evidence_id: str
    before_digest: str
    after_digest: str
    delta_digest: str
    reason: str

    def as_payload(self) -> dict[str, object]:
        return {
            "after_digest": self.after_digest,
            "before_digest": self.before_digest,
            "delta_digest": self.delta_digest,
            "evidence_id": self.evidence_id,
            "reason": self.reason,
            "source_layer": self.source_layer.value,
            "target_layer": self.target_layer.value,
        }


@dataclass(frozen=True)
class L2EpisodeAnchor:
    """Tamper-evident anchor for a closed L2 working episode.

    The anchor stores only digests and bounded counters.  Raw facts remain in the
    protected trace and semantic indexes; L2 working state is zeroed after
    consolidation so the next episode cannot inherit old phase mass.
    """

    episode_id: str
    episode_index: int
    anchor_hash: str
    clean_digest: str
    raw_digest: str
    tag_digest: str
    fact_count: int
    active_count: int
    promoted_rules_count: int
    reset_evidence_id: str

    def as_payload(self) -> dict[str, object]:
        return {
            "active_count": self.active_count,
            "anchor_hash": self.anchor_hash,
            "clean_digest": self.clean_digest,
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "fact_count": self.fact_count,
            "promoted_rules_count": self.promoted_rules_count,
            "raw_digest": self.raw_digest,
            "reset_evidence_id": self.reset_evidence_id,
            "tag_digest": self.tag_digest,
        }


@dataclass(frozen=True)
class L2WorkingContribution:
    """One active same-key contribution represented in the current L2 episode."""

    key: str
    semantic_delta: tuple[int, ...]
    tag_delta: tuple[int, ...]
    tagged_delta: tuple[int, ...]
    evidence_id: str
    weight: int
    fact_digest: str

    def as_payload(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "fact_digest": self.fact_digest,
            "key": self.key,
            "semantic_delta": self.semantic_delta,
            "semantic_delta_digest": active_state_digest({"delta": self.semantic_delta}),
            "tag_delta": self.tag_delta,
            "tag_delta_digest": active_state_digest({"delta": self.tag_delta}),
            "tagged_delta": self.tagged_delta,
            "tagged_delta_digest": active_state_digest({"delta": self.tagged_delta}),
            "weight": self.weight,
        }


@dataclass(frozen=True)
class L2CommitPreview:
    """Exact preview of a separated L2 working-memory commit."""

    key: str
    clean_before: tuple[int, ...]
    clean_after: tuple[int, ...]
    raw_before: tuple[int, ...]
    raw_after: tuple[int, ...]
    net_raw_delta: tuple[int, ...]
    net_clean_delta: tuple[int, ...]
    tag_accumulator_before: tuple[int, ...]
    tag_accumulator_after: tuple[int, ...]
    replaced_existing: bool
    episode_index: int
    fact_count_after: int

    def as_payload(self) -> dict[str, object]:
        return {
            "clean_after_digest": active_state_digest({"vector": self.clean_after}),
            "clean_before_digest": active_state_digest({"vector": self.clean_before}),
            "episode_index": self.episode_index,
            "fact_count_after": self.fact_count_after,
            "key": self.key,
            "net_clean_delta_digest": active_state_digest({"delta": self.net_clean_delta}),
            "net_raw_delta_digest": active_state_digest({"delta": self.net_raw_delta}),
            "raw_after_digest": active_state_digest({"vector": self.raw_after}),
            "raw_before_digest": active_state_digest({"vector": self.raw_before}),
            "replaced_existing": self.replaced_existing,
            "tag_accumulator_after_digest": active_state_digest({"vector": self.tag_accumulator_after}),
            "tag_accumulator_before_digest": active_state_digest({"vector": self.tag_accumulator_before}),
        }


class L123Body:
    """Three-level toroidal body: L1 sensory, L2 episodic, L3 semantic."""

    def __init__(self, *, dimension: int = DEFAULT_TORUS_DIMENSION, modulus: int = DEFAULT_MODULUS) -> None:
        self.dimension = int(dimension)
        self.modulus = int(modulus)
        self.projection = InterLevelProjection(dimension=dimension, modulus=modulus)
        self.l1 = LayerState.zero(LayerName.L1, dimension=dimension, modulus=modulus)
        self.l2 = LayerState.zero(LayerName.L2, dimension=dimension, modulus=modulus)
        self.l3 = LayerState.zero(LayerName.L3, dimension=dimension, modulus=modulus)
        self.l2_episode_index = 0
        self.l2_episode_phase = self._generate_l2_episode_phase(self.l2_episode_index)
        self.l2_episode_tag_accumulator = tuple(0 for _ in range(dimension))
        self.l2_episode_fact_count = 0
        self.l2_active_contributions: dict[str, L2WorkingContribution] = {}
        self.l2_archived_anchors: list[L2EpisodeAnchor] = []
        self.transitions: list[BodyTransition] = []

    def state(self, layer: LayerName | str) -> LayerState:
        name = layer if isinstance(layer, LayerName) else LayerName(str(layer))
        if name == LayerName.L1:
            return self.l1
        if name == LayerName.L2:
            return self.l2
        if name == LayerName.L3:
            return self.l3
        raise LayerError(f"unknown layer: {layer}")

    def apply_delta(self, state: LayerState, delta: LayerDelta) -> LayerState:
        source = delta.source_layer or state.name
        projected = self.projection.project(delta.delta, source=source, target=delta.target_layer)
        return state.apply(projected, evidence_id=delta.evidence_id, metadata={"reason": delta.reason})

    def apply_to_layer(self, delta: LayerDelta) -> BodyTransition:
        target = delta.target_layer
        before = self.state(target)
        after = self.apply_delta(before, delta)
        if target == LayerName.L1:
            self.l1 = after
        elif target == LayerName.L2:
            self.l2 = after
        elif target == LayerName.L3:
            self.l3 = after
        transition = BodyTransition(
            source_layer=delta.source_layer or target,
            target_layer=target,
            evidence_id=delta.evidence_id,
            before_digest=before.digest,
            after_digest=after.digest,
            delta_digest=active_state_digest({"delta": delta.delta, "target": target.value}),
            reason=delta.reason,
        )
        self.transitions.append(transition)
        return transition

    def observe_l1(self, text_or_id: str, *, evidence_id: str) -> BodyTransition:
        delta = hash_to_phase(f"l1:{text_or_id}:{evidence_id}", dimension=self.dimension, modulus=self.modulus, namespace="l1_observation")
        return self.apply_to_layer(LayerDelta(LayerName.L1, delta, evidence_id, source_layer=LayerName.L1, reason="l1_observation"))

    def observe_l1_phase(
        self,
        observed_phase: Iterable[int],
        *,
        evidence_id: str,
        reason: str = "l1_exact_sensory_projection",
    ) -> BodyTransition:
        """Set L1 to an already encoded observation phase via a toroidal delta.

        The sensory encoder computes ``u_obs``.  The body stores transitions, so
        it commits ``Delta_L1 = u_obs - h_L1 mod N``.  After this transition,
        ``self.l1.vector == u_obs`` exactly.
        """
        observed = tuple(q_mod(v, self.modulus) for v in observed_phase)
        if len(observed) != self.dimension:
            raise LayerError("L1 observed phase dimension mismatch")
        delta = tuple(q_sub(obs, cur, self.modulus) for cur, obs in zip(self.l1.vector, observed))
        return self.apply_to_layer(
            LayerDelta(
                LayerName.L1,
                delta,
                evidence_id,
                source_layer=LayerName.L1,
                reason=reason,
                modulus=self.modulus,
            )
        )

    def observe_l1_encoded(self, encoded: object, *, evidence_id: str | None = None) -> BodyTransition:
        """Commit an exact L1 encoder output without granting L2/L3 authority."""
        quantized = _observation_attr(encoded, "quantized", None)
        obs_evidence = evidence_id or str(getattr(quantized, "evidence_id", "l1_exact_sensory_projection"))
        observed_phase = tuple(_observation_attr(encoded, "observed_phase", ()))
        return self.observe_l1_phase(observed_phase, evidence_id=obs_evidence, reason="l1_exact_sensory_projection")

    def observe_simulated(self, observation: object, *, evidence_id: str | None = None) -> BodyTransition:
        """Commit a simulation-only sensory observation to L1.

        This is not a fact commit. It updates only the L1 toroidal surface and
        preserves the invariant real_sensor_commit_allowed = 0.
        """
        obs_evidence = evidence_id or str(_observation_attr(observation, "evidence_id", "simulated_observation"))
        delta = simulated_observation_delta(observation, dimension=self.dimension, modulus=self.modulus)
        modality = str(_observation_attr(observation, "modality", "simulated"))
        return self.apply_to_layer(
            LayerDelta(
                LayerName.L1,
                delta,
                obs_evidence,
                source_layer=LayerName.L1,
                reason=f"l1_simulated_observation:{modality}",
            )
        )

    def _l2_fact_key(self, item: FactDelta) -> str:
        return f"{item.fact.subject.value}\0{item.fact.relation.value}"

    def _generate_l2_episode_phase(self, episode_index: int) -> tuple[int, ...]:
        """Generate the deterministic Q256 phase tag for one L2 episode."""
        return hash_to_phase(
            f"l2-episode:{episode_index}",
            dimension=self.dimension,
            modulus=self.modulus,
            namespace="l2_episode_phase_tag",
        )

    def weighted_fact_delta(self, item: FactDelta) -> tuple[int, ...]:
        """Return the semantic ``w_f * Delta_f mod N`` contribution.

        This is the clean semantic part.  The raw L2 working torus stores the
        tagged contribution ``w_f*Delta_f + w_f*tau_episode`` and separately
        tracks the tag accumulator so clean reads are exact.
        """
        if len(item.delta) != self.dimension:
            raise LayerError("fact delta dimension mismatch for L2")
        return tuple(q_mod(int(value) * int(item.weight), item.modulus) for value in item.delta)

    def _weighted_l2_episode_tag(self, weight: int) -> tuple[int, ...]:
        return tuple(q_mod(int(value) * int(weight), self.modulus) for value in self.l2_episode_phase)

    def _l2_contribution_for_fact(self, item: FactDelta) -> L2WorkingContribution:
        semantic_delta = self.weighted_fact_delta(item)
        tag_delta = self._weighted_l2_episode_tag(item.weight)
        tagged_delta = q_vector_add(semantic_delta, tag_delta, self.modulus)
        return L2WorkingContribution(
            key=self._l2_fact_key(item),
            semantic_delta=semantic_delta,
            tag_delta=tag_delta,
            tagged_delta=tagged_delta,
            evidence_id=item.fact.evidence.value,
            weight=int(item.weight),
            fact_digest=active_state_digest({
                "delta": item.delta,
                "evidence": item.fact.evidence.value,
                "object": item.fact.object.value,
                "relation": item.fact.relation.value,
                "subject": item.fact.subject.value,
                "weight": item.weight,
            }),
        )

    def l2_clean_vector(self) -> tuple[int, ...]:
        """Return the active semantic L2 state with episode tags unbound.

        The mathematically exact unbinding is ``w_raw - sum_k(w_k*tau_ep)``.
        Subtracting only one episode tag would be correct for one fact but wrong
        for multi-fact episodes and weighted commits.
        """
        return q_vector_sub(self.l2.vector, self.l2_episode_tag_accumulator, self.modulus)

    def preview_l2_fact_commit(self, item: FactDelta) -> L2CommitPreview:
        """Preview active L2 update including same-key residual reversal."""
        new_contribution = self._l2_contribution_for_fact(item)
        old_contribution = self.l2_active_contributions.get(new_contribution.key)
        net_raw_delta = new_contribution.tagged_delta
        net_clean_delta = new_contribution.semantic_delta
        tag_after = q_vector_add(self.l2_episode_tag_accumulator, new_contribution.tag_delta, self.modulus)
        fact_count_after = self.l2_episode_fact_count + 1
        if old_contribution is not None:
            net_raw_delta = q_vector_sub(net_raw_delta, old_contribution.tagged_delta, self.modulus)
            net_clean_delta = q_vector_sub(net_clean_delta, old_contribution.semantic_delta, self.modulus)
            tag_after = q_vector_sub(tag_after, old_contribution.tag_delta, self.modulus)
            fact_count_after = self.l2_episode_fact_count
        raw_after = q_vector_add(self.l2.vector, net_raw_delta, self.modulus)
        clean_before = self.l2_clean_vector()
        clean_after = q_vector_sub(raw_after, tag_after, self.modulus)
        return L2CommitPreview(
            key=new_contribution.key,
            clean_before=clean_before,
            clean_after=clean_after,
            raw_before=self.l2.vector,
            raw_after=raw_after,
            net_raw_delta=net_raw_delta,
            net_clean_delta=net_clean_delta,
            tag_accumulator_before=self.l2_episode_tag_accumulator,
            tag_accumulator_after=tag_after,
            replaced_existing=old_contribution is not None,
            episode_index=self.l2_episode_index,
            fact_count_after=fact_count_after,
        )

    def commit_l2_fact(self, item: FactDelta, *, reason: str = "l2_fact_commit") -> BodyTransition:
        """Commit a fact to separated L2 working memory.

        Raw L2 receives a tagged working contribution.  Same-key supersession is
        handled by subtracting the previous active contribution from the working
        torus before adding the new one, so active L2 never becomes a lifetime
        accumulator.
        """
        preview = self.preview_l2_fact_commit(item)
        transition = self.apply_to_layer(
            LayerDelta(
                LayerName.L2,
                preview.net_raw_delta,
                item.fact.evidence.value,
                source_layer=LayerName.L2,
                reason=reason,
                modulus=item.modulus,
            )
        )
        contribution = self._l2_contribution_for_fact(item)
        self.l2_active_contributions[contribution.key] = contribution
        self.l2_episode_tag_accumulator = preview.tag_accumulator_after
        self.l2_episode_fact_count = preview.fact_count_after
        return transition

    def consolidate_l2_episode(
        self,
        *,
        episode_id: str,
        promoted_rules_count: int,
        evidence_id: str,
        reason: str = "l2_episode_consolidation_reset",
    ) -> tuple[BodyTransition, L2EpisodeAnchor]:
        """Anchor, archive and zero the L2 working episode.

        This implements the reset axiom: after consolidation, raw L2 is exactly
        zero, the tag accumulator is zero, active working contributions are
        cleared, and the next episode gets a fresh deterministic Q256 tag.
        """
        if not episode_id:
            raise LayerError("episode_id must be non-empty")
        clean = self.l2_clean_vector()
        payload = {
            "active_count": len(self.l2_active_contributions),
            "clean_state": clean,
            "episode_id": episode_id,
            "episode_index": self.l2_episode_index,
            "episode_phase": self.l2_episode_phase,
            "fact_count": self.l2_episode_fact_count,
            "promoted_rules_count": int(promoted_rules_count),
            "raw_state": self.l2.vector,
            "tag_accumulator": self.l2_episode_tag_accumulator,
        }
        anchor = L2EpisodeAnchor(
            episode_id=str(episode_id),
            episode_index=self.l2_episode_index,
            anchor_hash=active_state_digest(payload),
            clean_digest=active_state_digest({"vector": clean}),
            raw_digest=active_state_digest({"vector": self.l2.vector}),
            tag_digest=active_state_digest({"vector": self.l2_episode_tag_accumulator}),
            fact_count=self.l2_episode_fact_count,
            active_count=len(self.l2_active_contributions),
            promoted_rules_count=int(promoted_rules_count),
            reset_evidence_id=str(evidence_id),
        )
        reset_delta = tuple(q_sub(0, value, self.modulus) for value in self.l2.vector)
        transition = self.apply_to_layer(
            LayerDelta(
                LayerName.L2,
                reset_delta,
                evidence_id,
                source_layer=LayerName.L2,
                reason=reason,
                modulus=self.modulus,
            )
        )
        self.l2_archived_anchors.append(anchor)
        self.l2_episode_index += 1
        self.l2_episode_phase = self._generate_l2_episode_phase(self.l2_episode_index)
        self.l2_episode_tag_accumulator = tuple(0 for _ in range(self.dimension))
        self.l2_episode_fact_count = 0
        self.l2_active_contributions = {}
        return transition, anchor

    def promote_l3_rule(self, rule_id: str, *, evidence_id: str) -> BodyTransition:
        delta = hash_to_phase(f"l3-rule:{rule_id}", dimension=self.dimension, modulus=self.modulus, namespace="l3_rule")
        return self.apply_to_layer(LayerDelta(LayerName.L3, delta, evidence_id, source_layer=LayerName.L2, reason="l3_rule_promotion"))

    def commit_l3_semantic_state(
        self,
        semantic_state: Iterable[int] | object,
        *,
        evidence_id: str,
        reason: str = "l3_semantic_state_commit",
    ) -> BodyTransition:
        """Commit an externally consolidated L3 semantic state as a toroidal delta.

        The sleep/deduction organs may produce an L3 state, but they do not get
        direct authority over the body.  This method accepts only an already
        quantized integer state with exactly the body dimension and commits
        ``Delta_L3 = h3_target - h3_current mod N`` under the usual body trace.
        """
        phases_obj = getattr(semantic_state, "phases", semantic_state)
        target = tuple(q_mod(value, self.modulus) for value in phases_obj)
        if len(target) != self.dimension:
            raise LayerError("L3 semantic state dimension mismatch")
        delta = tuple(q_sub(dst, src, self.modulus) for src, dst in zip(self.l3.vector, target))
        return self.apply_to_layer(
            LayerDelta(
                LayerName.L3,
                delta,
                evidence_id,
                source_layer=LayerName.L3,
                reason=reason,
                modulus=self.modulus,
            )
        )

    def digest(self) -> str:
        return active_state_digest(self.snapshot())

    def snapshot(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "layers": {
                "L1": {"clock": self.l1.clock, "digest": self.l1.digest, "metadata": dict(self.l1.metadata), "vector": self.l1.vector},
                "L2": {"clock": self.l2.clock, "digest": self.l2.digest, "metadata": dict(self.l2.metadata), "vector": self.l2.vector},
                "L3": {"clock": self.l3.clock, "digest": self.l3.digest, "metadata": dict(self.l3.metadata), "vector": self.l3.vector},
            },
            "l2_working_memory": {
                "active_contributions": [item.as_payload() for item in sorted(self.l2_active_contributions.values(), key=lambda item: item.key)],
                "anchor_count": len(self.l2_archived_anchors),
                "anchors": [anchor.as_payload() for anchor in self.l2_archived_anchors],
                "clean_vector": self.l2_clean_vector(),
                "episode_fact_count": self.l2_episode_fact_count,
                "episode_index": self.l2_episode_index,
                "episode_phase": self.l2_episode_phase,
                "tag_accumulator": self.l2_episode_tag_accumulator,
            },
            "modulus": self.modulus,
            "transition_count": len(self.transitions),
        }

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, object]) -> "L123Body":
        """Restore L1/L2/L3 layer clocks/vectors/digests from a runtime snapshot.

        Snapshot restore is deterministic and does not replay AIR.  The stored
        layer digests are retained so that ``body.digest()`` round-trips exactly
        for a verified snapshot payload.  Transition objects are represented only
        by count because protected snapshots already contain the trace chain.
        """
        dimension = int(payload.get("dimension", DEFAULT_TORUS_DIMENSION))
        modulus = int(payload.get("modulus", DEFAULT_MODULUS))
        body = cls(dimension=dimension, modulus=modulus)
        layers = payload.get("layers", {})
        if not isinstance(layers, Mapping):
            raise LayerError("body snapshot layers must be a mapping")
        for name, attr in ((LayerName.L1, "l1"), (LayerName.L2, "l2"), (LayerName.L3, "l3")):
            layer_payload = layers.get(name.value)
            if not isinstance(layer_payload, Mapping):
                raise LayerError(f"missing layer snapshot for {name.value}")
            restored = LayerState(
                name=name,
                vector=tuple(int(v) for v in layer_payload.get("vector", ())),
                clock=int(layer_payload.get("clock", 0)),
                metadata=dict(layer_payload.get("metadata", {})) if isinstance(layer_payload.get("metadata", {}), Mapping) else {},
                modulus=modulus,
                digest=str(layer_payload.get("digest", "")),
            )
            setattr(body, attr, restored)
        working = payload.get("l2_working_memory", {})
        if isinstance(working, Mapping):
            body.l2_episode_index = int(working.get("episode_index", 0))
            raw_phase = working.get("episode_phase", ())
            if raw_phase:
                body.l2_episode_phase = tuple(q_mod(int(v), modulus) for v in raw_phase)  # type: ignore[arg-type]
            else:
                body.l2_episode_phase = body._generate_l2_episode_phase(body.l2_episode_index)
            raw_tag = working.get("tag_accumulator", ())
            if raw_tag:
                body.l2_episode_tag_accumulator = tuple(q_mod(int(v), modulus) for v in raw_tag)  # type: ignore[arg-type]
            body.l2_episode_fact_count = int(working.get("episode_fact_count", 0))
            body.l2_archived_anchors = []
            for raw_anchor in working.get("anchors", ()):  # type: ignore[union-attr]
                if isinstance(raw_anchor, Mapping):
                    body.l2_archived_anchors.append(L2EpisodeAnchor(
                        episode_id=str(raw_anchor.get("episode_id", "restored")),
                        episode_index=int(raw_anchor.get("episode_index", 0)),
                        anchor_hash=str(raw_anchor.get("anchor_hash", "")),
                        clean_digest=str(raw_anchor.get("clean_digest", "")),
                        raw_digest=str(raw_anchor.get("raw_digest", "")),
                        tag_digest=str(raw_anchor.get("tag_digest", "")),
                        fact_count=int(raw_anchor.get("fact_count", 0)),
                        active_count=int(raw_anchor.get("active_count", 0)),
                        promoted_rules_count=int(raw_anchor.get("promoted_rules_count", 0)),
                        reset_evidence_id=str(raw_anchor.get("reset_evidence_id", "restored_snapshot")),
                    ))
            body.l2_active_contributions = {}
            for raw_contrib in working.get("active_contributions", ()):  # type: ignore[union-attr]
                if isinstance(raw_contrib, Mapping):
                    key = str(raw_contrib.get("key", ""))
                    if key:
                        body.l2_active_contributions[key] = L2WorkingContribution(
                            key=key,
                            semantic_delta=tuple(q_mod(int(v), modulus) for v in raw_contrib.get("semantic_delta", ())),
                            tag_delta=tuple(q_mod(int(v), modulus) for v in raw_contrib.get("tag_delta", ())),
                            tagged_delta=tuple(q_mod(int(v), modulus) for v in raw_contrib.get("tagged_delta", ())),
                            evidence_id=str(raw_contrib.get("evidence_id", "restored_snapshot")),
                            weight=int(raw_contrib.get("weight", 1)),
                            fact_digest=str(raw_contrib.get("fact_digest", "restored")),
                        )
        transition_count = int(payload.get("transition_count", 0))
        if transition_count < 0:
            raise LayerError("transition_count must be non-negative")
        dummy = BodyTransition(
            source_layer=LayerName.L1,
            target_layer=LayerName.L1,
            evidence_id="restored_snapshot",
            before_digest="restored",
            after_digest="restored",
            delta_digest="restored",
            reason="restored_transition_count_reference",
        )
        body.transitions = [dummy for _ in range(transition_count)]
        return body
