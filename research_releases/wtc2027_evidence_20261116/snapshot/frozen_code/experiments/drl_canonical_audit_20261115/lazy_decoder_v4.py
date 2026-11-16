"""Install proposal-and-certify decoding for hard-four admission version 4."""

from __future__ import annotations

import inspect
import textwrap


VERSION = 4


def install() -> None:
    import torch
    from nets.attention_model import AttentionModel

    if getattr(AttentionModel, "_lazy_certificate_decoder_version", None) == VERSION:
        return

    original_get_log_p = AttentionModel._get_log_p
    original_inner = AttentionModel._inner

    def get_log_p_with_state(self, fixed, state, *args, **kwargs):
        result = original_get_log_p(self, fixed, state, *args, **kwargs)
        self._lazy_certificate_state = state
        return result

    def select_with_certificates(self, log_p, mask):
        state = self._lazy_certificate_state
        if getattr(state, "_hard4_admission_version", None) != VERSION:
            raise RuntimeError("Lazy decoder requires hard-four admission version 4")
        work_log_p = log_p
        work_mask = mask.clone()
        batch = log_p.size(0)
        selected = torch.zeros(batch, dtype=torch.long, device=log_p.device)
        joint_log_p = torch.zeros(batch, dtype=log_p.dtype, device=log_p.device)
        active = torch.ones(batch, dtype=torch.bool, device=log_p.device)
        max_proposals = log_p.size(1)
        proposals = 0
        while bool(active.any()):
            proposals += 1
            if proposals > max_proposals:
                raise RuntimeError("No certified action after exhausting proposal set")
            rows = active.nonzero(as_tuple=False).flatten()
            row_log_p = work_log_p[rows]
            if self.decode_type == "greedy":
                proposal = row_log_p.argmax(dim=1)
            elif self.decode_type == "sampling":
                proposal = torch.distributions.Categorical(logits=row_log_p).sample()
            else:
                raise RuntimeError(f"Unsupported decode type: {self.decode_type}")
            proposal_log_p = row_log_p.gather(1, proposal[:, None]).squeeze(1)
            if not torch.isfinite(proposal_log_p).all():
                raise RuntimeError("Proposal distribution exhausted before certification")
            joint_log_p = joint_log_p.index_add(0, rows, proposal_log_p)
            rejected_rows = []
            rejected_nodes = []
            for local, row in enumerate(rows.tolist()):
                node = int(proposal[local])
                if state._hard4_certify_one(row, node, work_mask[:, None, :]):
                    selected[row] = node
                    active[row] = False
                else:
                    rejected_rows.append(row)
                    rejected_nodes.append(node)
            if rejected_rows:
                next_mask = work_mask.clone()
                next_mask[rejected_rows, rejected_nodes] = True
                if bool(next_mask[rejected_rows].all(dim=1).any()):
                    raise RuntimeError("Hard-four oracle rejected every action")
                work_mask = next_mask
                work_log_p = torch.log_softmax(
                    log_p.masked_fill(work_mask, float("-inf")), dim=-1
                )
        self._lazy_certificate_joint_log_p = joint_log_p
        exceptional = [
            row
            for row, node in enumerate(selected.tolist())
            if node in (0, state.reject_index)
        ]
        if exceptional:
            work_mask = state._hard4_materialize_full(
                work_mask[:, None, :], exceptional
            )[:, 0, :]
        self._lazy_certificate_final_mask = work_mask
        return selected

    source = textwrap.dedent(inspect.getsource(original_inner))
    needle = "        selected = self._select_node(log_p[:, 0, :], mask[:, 0, :])\n"
    replacement = needle + (
        "        if hasattr(self, '_lazy_certificate_joint_log_p'):\n"
        "            mask = self._lazy_certificate_final_mask[:, None, :]\n"
        "            log_p = log_p.clone()\n"
        "            rows = torch.arange(log_p.size(0), device=log_p.device)\n"
        "            log_p[rows, 0, selected] = self._lazy_certificate_joint_log_p\n"
    )
    if source.count(needle) != 1:
        raise RuntimeError("Unexpected frozen decoder selection site")
    source = source.replace(needle, replacement)
    namespace = dict(original_inner.__globals__)
    exec(compile(source, "<lazy_certificate_inner_v4>", "exec"), namespace)

    AttentionModel._get_log_p = get_log_p_with_state
    AttentionModel._select_node = select_with_certificates
    AttentionModel._inner = namespace["_inner"]
    AttentionModel._lazy_certificate_decoder_version = VERSION

