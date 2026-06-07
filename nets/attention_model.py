import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many

from problem_mcvrptw_v2 import Config


REJECT_ACTION_EMBED_SCALE = 0.1


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 max_decode_steps=None,
                 max_consecutive_depot=8):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'
        self.is_mvrptw = problem.NAME in ('mvrptw', 'mcvrppdtw')

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.max_decode_steps = max_decode_steps
        self.max_consecutive_depot = max_consecutive_depot

        if self.is_vrp or self.is_orienteering or self.is_pctsp or self.is_mvrptw:
            if self.is_mvrptw:
                step_context_dim = embedding_dim + 4
                node_dim = 7
            else:
                step_context_dim = embedding_dim + 1
                if self.is_pctsp:
                    node_dim = 4
                else:
                    node_dim = 3

            self.init_embed_depot = nn.Linear(2, embedding_dim)

            if self.is_vrp and self.allow_partial:
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        else:
            assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim
            node_dim = 2
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.type_embedding = nn.Embedding(5, embedding_dim) if self.is_mvrptw else None
        self.numeric_proj = nn.Linear(node_dim - 1, embedding_dim) if self.is_mvrptw else None

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
            feed_forward_hidden=hidden_dim,
        )

        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        self.reject_proj = nn.Linear(step_context_dim, 1)
        assert embedding_dim % n_heads == 0
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def _build_pd_pair_mask(self, input_data):
        if not self.is_mvrptw:
            return None

        if 'pd_pair_mask' in input_data:
            return input_data['pd_pair_mask'].to(dtype=torch.bool)

        node_type = input_data['node_type'].long()
        batch_size, n_loc = node_type.size()
        device = node_type.device
        n_orders = n_loc // 2

        pd_pair_mask = torch.zeros(batch_size, n_loc + 1, n_loc + 1, dtype=torch.bool, device=device)
        if n_orders == 0:
            return pd_pair_mask

        order_idx = torch.arange(n_orders, device=device)
        pickup_nodes = order_idx + 1
        delivery_nodes = order_idx + n_orders + 1

        passenger_pickup = node_type[:, :n_orders] == 1
        passenger_delivery = node_type[:, n_orders:] == 1
        cargo_pickup = node_type[:, :n_orders] == 0
        cargo_delivery = node_type[:, n_orders:] == 0

        pd_pair_mask[:, pickup_nodes, delivery_nodes] = passenger_pickup | cargo_pickup
        pd_pair_mask[:, delivery_nodes, pickup_nodes] = passenger_delivery | cargo_delivery
        return pd_pair_mask

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        init_embed = self._init_embed(input)
        pd_pair_mask = self._build_pd_pair_mask(input)

        if self.checkpoint_encoder and self.training:
            embeddings, _ = checkpoint(
                lambda embed_input, pair_mask: self.embedder(embed_input, pd_pair_mask=pair_mask),
                init_embed,
                pd_pair_mask,
                use_reentrant=False,
            )
        else:
            embeddings, _ = self.embedder(init_embed, pd_pair_mask=pd_pair_mask)

        _log_p, pi = self._inner(input, embeddings)

        cost, mask = self.problem.get_costs(input, pi)
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input), pd_pair_mask=self._build_pd_pair_mask(input))
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        return log_p.sum(1)

    def _map_node_types(self, input_data):
        node_type = input_data['node_type'].long()
        n_orders = node_type.size(1) // 2
        mapped = torch.zeros_like(node_type, dtype=torch.long)
        if n_orders == 0:
            return mapped
        pickup_is_passenger = input_data['demand_passenger'][:, :n_orders].abs() > 1e-8
        delivery_is_passenger = input_data['demand_passenger'][:, n_orders:].abs() > 1e-8
        mapped[:, :n_orders] = torch.where(pickup_is_passenger, 1, 3)
        mapped[:, n_orders:] = torch.where(delivery_is_passenger, 2, 4)
        return mapped

    def _init_embed(self, input):

        if self.is_mvrptw:
            node_type_ids = self._map_node_types(input)
            numeric = torch.cat(
                (
                    input['loc'],
                    input['demand_passenger'][:, :, None],
                    input['demand_cargo'][:, :, None],
                    input['time_windows']
                ),
                -1
            )
            h_loc = self.numeric_proj(numeric) + self.type_embedding(node_type_ids)
            depot_embed = self.init_embed_depot(input['depot'])[:, None, :] + self.type_embedding.weight[0][None, None, :]
            return torch.cat((depot_embed, h_loc), 1)
        elif self.is_vrp or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand', )
            elif self.is_orienteering:
                features = ('prize', )
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )
        return self.init_embed(input)

    def _inner(self, input, embeddings):
        outputs = []
        sequences = []

        state = self.problem.make_state(input)
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)
        node_count = embeddings.size(1)
        max_steps = self.max_decode_steps or max(node_count * 3, 8)
        consecutive_depot = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device)

        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            if i >= max_steps:
                break

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    state = state[unfinished]
                    fixed = fixed[unfinished]
                    consecutive_depot = consecutive_depot[unfinished]

            log_p, mask = self._get_log_p(fixed, state, consecutive_depot=consecutive_depot)
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])
            consecutive_depot = torch.where(selected == 0, consecutive_depot + 1, torch.zeros_like(consecutive_depot))

            state = state.update(selected)

            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_, consecutive_depot_ = log_p, selected, consecutive_depot
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)
                consecutive_depot = consecutive_depot_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_
                consecutive_depot[state.ids[:, 0]] = consecutive_depot_

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        if len(outputs) == 0:
            dummy_log_p = torch.full((batch_size, 1, node_count + 1), -math.inf, device=embeddings.device)
            dummy_log_p[:, :, 0] = 0.0
            dummy_selected = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device)
            outputs.append(dummy_log_p[:, 0, :])
            sequences.append(dummy_selected)

        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        return sample_many(
            lambda input: self._inner(*input),
            lambda input, pi: self.problem.get_costs(input[0], pi),
            (input, self.embedder(self._init_embed(input), pd_pair_mask=self._build_pd_pair_mask(input))[0]),
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True, consecutive_depot=None):
        step_context = self._get_parallel_step_context(fixed.node_embeddings, state)
        query = fixed.context_node_projected + self.project_step_context(step_context)

        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        mask = state.get_mask()
        if consecutive_depot is not None and self.max_consecutive_depot is not None:
            depot_limit_reached = (consecutive_depot >= self.max_consecutive_depot)
            if depot_limit_reached.any():
                state = state._replace(terminal_=state.terminal_ | depot_limit_reached[:, None])
                mask = mask.clone()
                mask[depot_limit_reached, :, :] = True
                mask[depot_limit_reached, :, 0] = False

        all_masked = mask[:, 0, :].all(-1)
        if all_masked.any():
            state = state._replace(terminal_=state.terminal_ | all_masked[:, None])
            mask = mask.clone()
            mask[all_masked, :, 0] = False

        log_p, glimpse = self._one_to_many_logits(query, step_context, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if self.is_mvrptw:
            operation_span = max(Config.OPERATION_END - Config.OPERATION_START, 1e-6)
            normalized_time = ((state.current_time - Config.OPERATION_START) / operation_span).clamp(0.0, 1.0)
            remaining_vehicle_budget = state.get_remaining_vehicle_budget()[:, :, None]
            if from_depot:
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        self.problem.PASSENGER_CAPACITY - torch.zeros_like(state.used_capacity_passenger[:, :, None]),
                        self.problem.CARGO_CAPACITY - torch.zeros_like(state.used_capacity_cargo[:, :, None]),
                        torch.zeros_like(state.current_time[:, :, None]),
                        remaining_vehicle_budget
                    ),
                    -1
                )
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous().view(batch_size, num_steps, 1).expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    self.problem.PASSENGER_CAPACITY - state.used_capacity_passenger[:, :, None],
                    self.problem.CARGO_CAPACITY - state.used_capacity_cargo[:, :, None],
                    normalized_time[:, :, None],
                    remaining_vehicle_budget
                ),
                -1
            )
        elif self.is_vrp:
            if from_depot:
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous().view(batch_size, num_steps, 1).expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]
                ),
                -1
            )
        elif self.is_orienteering or self.is_pctsp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous().view(batch_size, num_steps, 1).expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    (
                        state.get_remaining_length()[:, :, None]
                        if self.is_orienteering
                        else state.get_remaining_prize_to_collect()[:, :, None]
                    )
                ),
                -1
            )
        else:
            if num_steps == 1:
                if state.i.item() == 0:
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                return embeddings.gather(
                    1,
                    torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                ).view(batch_size, 1, -1)
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, step_context, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            node_mask = mask[:, :, :-1]
            compatibility[node_mask[:, :, None, :][None, :, :, :, :].expand_as(compatibility)] = -math.inf

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
        )

        final_Q = glimpse
        node_logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        reject_logit = self.reject_proj(step_context)
        logits = torch.cat((node_logits, reject_logit), dim=-1)

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):
        if self.is_vrp and self.allow_partial:
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )

        # Encoder-side attention tensors only cover depot + real nodes; synthetic reject action is appended later.
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )
