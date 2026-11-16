"""Lazy hard-four certificates using the unchanged version-3 oracle.

This module changes certificate *scheduling*, not the physical model, scalar
search, node budget, or float32 replay verification.  It deliberately exposes
one-candidate certification to the decoder and never certifies an action merely
because the bounded search returned a suffix: the suffix must replay through
the common State mask first.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import textwrap

import admission_targeted as reference


VERSION = 4
STATS = reference.STATS
DECISION_CACHE = {}
for _key in (
    "lazy_queries",
    "lazy_certified",
    "lazy_uncertified",
    "lazy_proposals",
):
    STATS.setdefault(_key, 0)


def clear() -> None:
    reference.clear()
    DECISION_CACHE.clear()
    for key in tuple(STATS):
        STATS[key] = 0


def install(eager: bool = False) -> None:
    import torch
    from state_mcvrptw_v2 import StateMCVRPPDTW as State

    if getattr(State, "_hard4_admission_version", None) == VERSION:
        return
    if not hasattr(State, "_before_completion_guard"):
        raise RuntimeError("Install corrected clock and completion guard first")

    # If v3 was installed earlier in the same diagnostic process, recover the
    # exact pre-admission common mask.  Otherwise the current mask is that base.
    base_mask_fn = getattr(State, "_before_hard4_admission", State.get_mask)
    raw = State._before_completion_guard
    code = textwrap.dedent(inspect.getsource(raw))
    before = (
        "def get_mask(self, return_debug=False, skip_pickup_commitment=False, "
        "benchmark_stats=None, return_urgency_slack=False, aux_outputs=None):"
    )
    after = (
        "def target_mask(self, target_node, return_debug=False, "
        "skip_pickup_commitment=False, benchmark_stats=None, "
        "return_urgency_slack=False, aux_outputs=None):"
    )
    if code.count(before) != 1:
        raise RuntimeError("Unexpected frozen mask signature")
    code = code.replace(before, after)
    anchor = "        pickup_physical_feasible = ~service_mask[:, :n_orders]"
    inserted = (
        anchor
        + "\n        selected_only = torch.zeros_like(pickup_physical_feasible)"
        + "\n        if 1 <= target_node <= n_orders:"
        + "\n            selected_only[:, target_node-1] = "
        + "pickup_physical_feasible[:, target_node-1]"
        + "\n        pickup_physical_feasible = selected_only"
    )
    if code.count(anchor) != 1:
        raise RuntimeError("Unexpected commitment filter")
    code = code.replace(anchor, inserted)
    generated = Path(__file__).resolve().parent / "generated_target_mask_v4.py"
    if generated.exists() and generated.read_text(encoding="utf-8") != code:
        raise RuntimeError("Refuse to overwrite a changed target-mask helper")
    generated.write_text(code, encoding="utf-8")
    namespace = dict(raw.__globals__)
    exec(compile(code, str(generated), "exec"), namespace)
    target_mask = namespace["target_mask"]

    def selected_mask(current, node):
        STATS["target_queries"] += 1
        mask = target_mask(current, node)
        if bool(mask[0, 0, node]) or (
            node in (0, current.reject_index)
            and bool(current.has_open_started_orders().any())
        ):
            STATS["target_full_fallbacks"] += 1
            mask = base_mask_fn(current)
        if reference.AUDIT_TARGET_QUERIES:
            full = base_mask_fn(current)
            if bool(mask[0, 0, node]) != bool(full[0, 0, node]):
                raise AssertionError("Target predicate differs from full common mask")
            STATS["target_verified"] += 1
        return mask

    def problem(state, batch_index):
        key = (id(state.coords), int(state.ids[batch_index]))
        if key not in reference.STATIC:
            reference.STATIC[key] = reference.Problem(state, batch_index)
        return reference.STATIC[key]

    def verify(state, sequence, prob):
        current = state
        for node in sequence:
            mask = selected_mask(current, node)
            if bool(mask[0, 0, node]):
                return False
            snap = prob.capture(current, 0)
            if prob.step(snap, node) is None:
                return False
            if node == 0:
                return True
            current = current.update(
                torch.tensor([node], device=mask.device), current_mask=mask
            )
        return False

    def certified(next_state, prob):
        snap = prob.capture(next_state, 0)
        key = (prob.token, snap)
        if key in DECISION_CACHE:
            STATS["cache_hits"] += 1
            return DECISION_CACHE[key]
        suffix = prob.search(snap, [0])
        if suffix is None:
            DECISION_CACHE[key] = False
            return False
        if not verify(next_state, suffix, prob):
            STATS["verification_failures"] += 1
            DECISION_CACHE[key] = False
            return False
        DECISION_CACHE[key] = True
        return True

    def guarded_base(self, *args, **kwargs):
        if kwargs.get("skip_pickup_commitment", False):
            return base_mask_fn(self, *args, **kwargs)
        result = base_mask_fn(self, *args, **kwargs)
        mask = result[0] if isinstance(result, tuple) else result
        mask = mask.clone()
        STATS["calls"] += 1
        finished = self.get_finished()
        for b in range(mask.shape[0]):
            if bool(finished[b]):
                continue
            one = self[b : b + 1]
            prob = problem(self, b)
            snap = prob.capture(one, 0)
            if snap.node != 0:
                mask[b, :, one.reject_index] = True
                if snap.completed < 4 or 1 in snap.status:
                    mask[b, :, 0] = True
                elif prob.step(snap, 0) is None:
                    mask[b, :, 0] = True
            else:
                mask[b, :, 0] = True
            if not (~mask[b]).any():
                STATS["fails_closed"] += 1
                raise RuntimeError("No base action remains under hard-four rules")
        return (mask, *result[1:]) if isinstance(result, tuple) else mask

    def certify_one(self, batch_index, node, current_mask):
        STATS["lazy_queries"] += 1
        STATS["lazy_proposals"] += 1
        node = int(node)
        if bool(current_mask[batch_index, 0, node]):
            STATS["lazy_uncertified"] += 1
            return False
        one = self[batch_index : batch_index + 1]
        if node in (0, one.reject_index):
            STATS["lazy_certified"] += 1
            return True
        STATS["candidates"] += 1
        prob = problem(self, batch_index)
        snap = prob.capture(one, 0)
        if prob.step(snap, node) is None:
            ok = False
        else:
            next_state = one.update(
                torch.tensor([node], device=current_mask.device),
                current_mask=current_mask[batch_index : batch_index + 1],
            )
            ok = certified(next_state, prob)
        if ok:
            STATS["certified"] += 1
            STATS["lazy_certified"] += 1
        else:
            STATS["uncertified"] += 1
            STATS["lazy_uncertified"] += 1
        return ok

    def materialize_full(self, current_mask, batch_indices=None):
        """Build the v3-equivalent mask only when State.update consumes it.

        Reject and depot actions inspect non-selected mask bits for deterministic
        rejection/deadlock bookkeeping.  Service updates do not.  Materializing
        on those exceptional actions preserves state semantics without paying
        the all-candidate cost at every service step.
        """
        full = current_mask.clone()
        indices = (
            range(full.shape[0]) if batch_indices is None else tuple(batch_indices)
        )
        for b in indices:
            candidates = (
                (~full[b, 0, 1 : 2 * self.n_orders + 1])
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
            for offset in candidates:
                node = int(offset + 1)
                if not certify_one(self, b, node, full):
                    full[b, :, node] = True
            if not (~full[b]).any():
                STATS["fails_closed"] += 1
                raise RuntimeError("No certified hard-four action")
        return full

    def guarded_eager(self, *args, **kwargs):
        result = guarded_base(self, *args, **kwargs)
        mask = result[0] if isinstance(result, tuple) else result
        full = materialize_full(self, mask)
        return (full, *result[1:]) if isinstance(result, tuple) else full

    State._before_hard4_admission = base_mask_fn
    State.get_mask = guarded_eager if eager else guarded_base
    State._hard4_certify_one = certify_one
    State._hard4_materialize_full = materialize_full
    State._hard4_admission_version = VERSION

