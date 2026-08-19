import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
import time
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many

from problem_mcvrptw_v2 import Config


REJECT_ACTION_EMBED_SCALE = 0.1


def _merge_benchmark_stats(base, extra):
    if base is None:
        return extra
    if extra is None:
        return base
    merged = {
        'seconds': dict(base.get('seconds', {})),
        'calls': dict(base.get('calls', {})),
    }
    for name, value in extra.get('seconds', {}).items():
        merged['seconds'][name] = merged['seconds'].get(name, 0.0) + float(value)
    for name, value in extra.get('calls', {}).items():
        merged['calls'][name] = merged['calls'].get(name, 0) + int(value)
    return merged


def set_decode_type(model, decode_type):
    if isinstance(model, (DataParallel, DistributedDataParallel)):
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
                 max_consecutive_depot=8,
                 reject_init_bias=-2.5,
                 decode_pickup_urgency_bias=0.0,
                 decode_pickup_urgency_horizon_hours=1.0):
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
        self.decode_pickup_urgency_bias = float(decode_pickup_urgency_bias)
        self.decode_pickup_urgency_horizon_hours = max(float(decode_pickup_urgency_horizon_hours), 1e-6)

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
        nn.init.constant_(self.reject_proj.bias, reject_init_bias)
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

    @staticmethod
    def _repeat_batch_for_pomo(input_data, pomo_size):
        if pomo_size <= 1:
            return input_data
        return {
            key: value.repeat_interleave(pomo_size, dim=0) if torch.is_tensor(value) else value
            for key, value in input_data.items()
        }

    @staticmethod
    def _build_logical_pomo_index(batch_size, pomo_size, device):
        return torch.arange(batch_size, device=device, dtype=torch.long).repeat_interleave(pomo_size)

    def forward(self, input, return_pi=False, state_kwargs=None, return_debug=False, return_benchmark=False, logical_pomo_size=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :param logical_pomo_size: decode each base instance with multiple logical rollouts while encoding the input once
        :return:
        """
        benchmark = {'seconds': {}, 'calls': {}} if state_kwargs and state_kwargs.get('benchmark_timing') else None
        state_init_kwargs = dict(state_kwargs or {})
        benchmark_enabled = bool(state_init_kwargs.pop('benchmark_timing', False))
        logical_pomo_size = max(int(logical_pomo_size or 1), 1)

        def _record(name, seconds):
            nonlocal benchmark
            if not benchmark_enabled:
                return
            if benchmark is None:
                benchmark = {'seconds': {}, 'calls': {}}
            benchmark['seconds'][name] = benchmark['seconds'].get(name, 0.0) + float(seconds)
            benchmark['calls'][name] = benchmark['calls'].get(name, 0) + 1

        model_input = input
        pomo_ids = None
        if logical_pomo_size > 1:
            batch_size = input['loc'].size(0)
            if batch_size <= 0:
                raise ValueError('logical_pomo_size requires a non-empty batch')
            pomo_ids = self._build_logical_pomo_index(batch_size, logical_pomo_size, input['loc'].device)
            model_input = self._repeat_batch_for_pomo(input, logical_pomo_size)

        init_embed_start = time.perf_counter() if benchmark_enabled else None
        init_embed = self._init_embed(input)
        pd_pair_mask = self._build_pd_pair_mask(input)
        if benchmark_enabled:
            _record('model_init_embed', time.perf_counter() - init_embed_start)

        encoder_start = time.perf_counter() if benchmark_enabled else None
        if self.checkpoint_encoder and self.training:
            embeddings, _ = checkpoint(
                lambda embed_input, pair_mask: self.embedder(embed_input, pd_pair_mask=pair_mask),
                init_embed,
                pd_pair_mask,
                use_reentrant=False,
            )
        else:
            embeddings, _ = self.embedder(init_embed, pd_pair_mask=pd_pair_mask)
        if logical_pomo_size > 1:
            embeddings = embeddings.index_select(0, pomo_ids)
        if benchmark_enabled:
            _record('model_encoder', time.perf_counter() - encoder_start)

        inner_start = time.perf_counter() if benchmark_enabled else None
        _log_p, pi, debug = self._inner(
            model_input,
            embeddings,
            state_kwargs=state_init_kwargs,
            return_debug=return_debug,
            benchmark_timing=benchmark_enabled,
        )
        if benchmark_enabled:
            _record('model_decode_inner', time.perf_counter() - inner_start)
            if isinstance(debug, dict):
                benchmark = _merge_benchmark_stats(benchmark, debug.pop('benchmark_timing', None))

        costs_start = time.perf_counter() if benchmark_enabled else None
        # logical_pomo_size > 1 时，model_input 已经 repeat 过，避免重复构造大张量
        cost_input = model_input
        cost, mask = self.problem.get_costs(cost_input, pi)
        if benchmark_enabled:
            _record('model_get_costs', time.perf_counter() - costs_start)

        ll_start = time.perf_counter() if benchmark_enabled else None
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if benchmark_enabled:
            _record('model_log_likelihood', time.perf_counter() - ll_start)

        benchmark_payload = benchmark or {'seconds': {}, 'calls': {}}
        if return_debug:
            if debug is None:
                debug = {}
            if benchmark_enabled:
                debug['benchmark_timing'] = benchmark_payload
        if return_pi and return_debug:
            result = (cost, ll, pi, debug)
        elif return_pi:
            result = (cost, ll, pi)
        elif return_debug:
            result = (cost, ll, debug)
        else:
            result = (cost, ll)

        if return_benchmark:
            return (*result, benchmark_payload)
        return result

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

    def _inner(self, input, embeddings, state_kwargs=None, return_debug=False, benchmark_timing=False):
        outputs = []
        sequences = []

        benchmark = {'seconds': {}, 'calls': {}} if benchmark_timing else None
        benchmark_enabled = bool(benchmark_timing)

        def _record(name, seconds):
            nonlocal benchmark
            if not benchmark_enabled:
                return
            if benchmark is None:
                benchmark = {'seconds': {}, 'calls': {}}
            benchmark['seconds'][name] = benchmark['seconds'].get(name, 0.0) + float(seconds)
            benchmark['calls'][name] = benchmark['calls'].get(name, 0) + 1

        state_start = time.perf_counter() if benchmark_enabled else None
        state = self.problem.make_state(input, **(state_kwargs or {}))
        if benchmark_enabled:
            _record('decode_make_state', time.perf_counter() - state_start)

        precompute_start = time.perf_counter() if benchmark_enabled else None
        fixed = self._precompute(embeddings)
        if benchmark_enabled:
            _record('decode_precompute_fixed', time.perf_counter() - precompute_start)

        batch_size = state.ids.size(0)
        node_count = embeddings.size(1)
        max_steps = self.max_decode_steps or max(node_count * 3, 8)
        consecutive_depot = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device)

        def _restore_to_full_batch(tensor, active_ids, fill_value=0):
            if tensor is None or tensor.size(0) == batch_size:
                return tensor
            restored = tensor.new_full((batch_size, *tensor.size()[1:]), fill_value)
            restored[active_ids] = tensor
            return restored

        debug_totals = None
        if return_debug:
            debug_totals = {
                'diag_steps': torch.zeros(batch_size, device=embeddings.device),
                'diag_feasible_pickups': torch.zeros(batch_size, device=embeddings.device),
                'diag_feasible_deliveries': torch.zeros(batch_size, device=embeddings.device),
                'diag_feasible_service': torch.zeros(batch_size, device=embeddings.device),
                'diag_any_service_feasible': torch.zeros(batch_size, device=embeddings.device),
                'diag_depot_only': torch.zeros(batch_size, device=embeddings.device),
                'diag_reject_available_rate': torch.zeros(batch_size, device=embeddings.device),
                'diag_service_feasible_but_selected_depot': torch.zeros(batch_size, device=embeddings.device),
                'diag_service_feasible_but_selected_reject': torch.zeros(batch_size, device=embeddings.device),
                'diag_selected_pickup': torch.zeros(batch_size, device=embeddings.device),
                'diag_selected_delivery': torch.zeros(batch_size, device=embeddings.device),
                'diag_selected_depot': torch.zeros(batch_size, device=embeddings.device),
                'diag_selected_reject': torch.zeros(batch_size, device=embeddings.device),
                'diag_mask_visited': torch.zeros(batch_size, device=embeddings.device),
                'diag_mask_precedence': torch.zeros(batch_size, device=embeddings.device),
                'diag_mask_cap_passenger': torch.zeros(batch_size, device=embeddings.device),
                'diag_mask_cap_cargo': torch.zeros(batch_size, device=embeddings.device),
                'diag_mask_pickup_tw': torch.zeros(batch_size, device=embeddings.device),
                'diag_mask_ride_time': torch.zeros(batch_size, device=embeddings.device),
                'diag_mask_trip_time': torch.zeros(batch_size, device=embeddings.device),
                'diag_mask_ops_end': torch.zeros(batch_size, device=embeddings.device),
                'diag_mask_pickup_commitment': torch.zeros(batch_size, device=embeddings.device),
                'diag_open_started_count': torch.zeros(batch_size, device=embeddings.device),
                'diag_open_started_eq2': torch.zeros(batch_size, device=embeddings.device),
                'diag_second_pickup_feasible': torch.zeros(batch_size, device=embeddings.device),
                'diag_second_pickup_blocked_by_commitment': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_k': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_completion': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_completion_ride_time': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_completion_trip_time': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_completion_ops_end': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_completion_open_over_6': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_completion_other': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_next_state': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_next_state_precedence': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_next_state_ride_time': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_next_state_trip_time': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_next_state_ops_end': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_next_state_delivery_viability': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_next_state_vehicle_limit': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_next_state_mixed': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_next_state_other': torch.zeros(batch_size, device=embeddings.device),
                'diag_pickup_commitment_block_by_fallback': torch.zeros(batch_size, device=embeddings.device),
                'diag_delivery_viability_masked': torch.zeros(batch_size, device=embeddings.device),
                'diag_delivery_viability_fallback': torch.zeros(batch_size, device=embeddings.device),
                'diag_mask_vehicle_limit': torch.zeros(batch_size, device=embeddings.device),
                'diag_depot_carry_block': torch.zeros(batch_size, device=embeddings.device),
                'diag_depot_no_work_block': torch.zeros(batch_size, device=embeddings.device),
                'diag_depot_fallback_used': torch.zeros(batch_size, device=embeddings.device),
                'diag_reject_candidate_available': torch.zeros(batch_size, device=embeddings.device),
                'diag_reject_allowed': torch.zeros(batch_size, device=embeddings.device),
                'diag_reject_predeparture_available': torch.zeros(batch_size, device=embeddings.device),
                'diag_reject_inroute_available': torch.zeros(batch_size, device=embeddings.device),
            }

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

            active_ids = state.ids[:, 0]
            get_log_p_start = time.perf_counter() if benchmark_enabled else None
            log_p, mask, step_debug = self._get_log_p(
                fixed,
                state,
                consecutive_depot=consecutive_depot,
                return_debug=return_debug,
                benchmark_stats=benchmark if benchmark_enabled else None,
            )
            if benchmark_enabled:
                _record('decode_get_log_p_total', time.perf_counter() - get_log_p_start)
            select_start = time.perf_counter() if benchmark_enabled else None
            selected = self._select_node(log_p[:, 0, :], mask[:, 0, :])
            if benchmark_enabled:
                _record('decode_select_node', time.perf_counter() - select_start)
            if return_debug:
                active_debug_updates = {
                    'diag_selected_depot': (selected == 0).float(),
                    'diag_selected_reject': (selected == state.reject_index).float(),
                    'diag_selected_pickup': ((selected >= 1) & (selected <= state.n_orders)).float(),
                    'diag_selected_delivery': ((selected >= state.n_orders + 1) & (selected <= 2 * state.n_orders)).float(),
                }
                service_feasible = step_debug['diag_any_service_feasible'] > 0
                active_debug_updates['diag_service_feasible_but_selected_depot'] = (
                    service_feasible & (selected == 0)
                ).float()
                active_debug_updates['diag_service_feasible_but_selected_reject'] = (
                    service_feasible & (selected == state.reject_index)
                ).float()
                for key in debug_totals:
                    if key in step_debug:
                        active_debug_updates[key] = step_debug[key]
                for key, value in active_debug_updates.items():
                    debug_totals[key] += _restore_to_full_batch(value, active_ids, fill_value=0)
            consecutive_depot = torch.where(selected == 0, consecutive_depot + 1, torch.zeros_like(consecutive_depot))

            update_start = time.perf_counter() if benchmark_enabled else None
            state = state.update(selected, current_mask=mask)
            if benchmark_enabled:
                _record('decode_state_update', time.perf_counter() - update_start)

            outputs.append(_restore_to_full_batch(log_p, active_ids, fill_value=0)[:, 0, :])
            sequences.append(_restore_to_full_batch(selected, active_ids, fill_value=0))

            i += 1

        if len(outputs) == 0:
            dummy_log_p = torch.full((batch_size, 1, node_count + 1), -math.inf, device=embeddings.device)
            dummy_log_p[:, :, 0] = 0.0
            dummy_selected = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device)
            outputs.append(dummy_log_p[:, 0, :])
            sequences.append(dummy_selected)

        if return_debug:
            denom = torch.clamp(debug_totals['diag_steps'], min=1.0)
            normalized_debug = {}
            rate_keys = {
                'diag_feasible_pickups',
                'diag_feasible_deliveries',
                'diag_feasible_service',
                'diag_any_service_feasible',
                'diag_depot_only',
                'diag_reject_available_rate',
                'diag_service_feasible_but_selected_depot',
                'diag_service_feasible_but_selected_reject',
                'diag_selected_pickup',
                'diag_selected_delivery',
                'diag_selected_depot',
                'diag_selected_reject',
                'diag_mask_visited',
                'diag_mask_precedence',
                'diag_mask_cap_passenger',
                'diag_mask_cap_cargo',
                'diag_mask_pickup_tw',
                'diag_mask_ride_time',
                'diag_mask_trip_time',
                'diag_mask_ops_end',
                'diag_mask_pickup_commitment',
                'diag_open_started_count',
                'diag_open_started_eq2',
                'diag_second_pickup_feasible',
                'diag_second_pickup_blocked_by_commitment',
                'diag_pickup_commitment_block_by_k',
                'diag_pickup_commitment_block_by_completion',
                'diag_pickup_commitment_block_by_completion_ride_time',
                'diag_pickup_commitment_block_by_completion_trip_time',
                'diag_pickup_commitment_block_by_completion_ops_end',
                'diag_pickup_commitment_block_by_completion_open_over_6',
                'diag_pickup_commitment_block_by_completion_other',
                'diag_pickup_commitment_block_by_next_state',
                'diag_pickup_commitment_block_by_next_state_precedence',
                'diag_pickup_commitment_block_by_next_state_ride_time',
                'diag_pickup_commitment_block_by_next_state_trip_time',
                'diag_pickup_commitment_block_by_next_state_ops_end',
                'diag_pickup_commitment_block_by_next_state_delivery_viability',
                'diag_pickup_commitment_block_by_next_state_vehicle_limit',
                'diag_pickup_commitment_block_by_next_state_mixed',
                'diag_pickup_commitment_block_by_next_state_other',
                'diag_pickup_commitment_block_by_fallback',
                'diag_delivery_viability_masked',
                'diag_delivery_viability_fallback',
                'diag_mask_vehicle_limit',
                'diag_depot_carry_block',
                'diag_depot_no_work_block',
                'diag_depot_fallback_used',
                'diag_reject_candidate_available',
                'diag_reject_allowed',
                'diag_reject_predeparture_available',
                'diag_reject_inroute_available',
            }
            for key, value in debug_totals.items():
                if key == 'diag_steps':
                    normalized_debug[key] = value
                elif key in rate_keys:
                    normalized_debug[key] = value / denom
                else:
                    normalized_debug[key] = value
            if benchmark_enabled:
                normalized_debug['benchmark_timing'] = benchmark or {'seconds': {}, 'calls': {}}
            return torch.stack(outputs, 1), torch.stack(sequences, 1), normalized_debug

        return torch.stack(outputs, 1), torch.stack(sequences, 1), {'benchmark_timing': benchmark or {'seconds': {}, 'calls': {}}} if benchmark_enabled else None

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        return sample_many(
            lambda input: self._inner(*input),
            lambda input, pi: self.problem.get_costs(input[0], pi),
            (input, self.embedder(self._init_embed(input), pd_pair_mask=self._build_pd_pair_mask(input))[0]),
            batch_rep, iter_rep
        )

    def _select_node(self, log_p, mask):
        assert (log_p == log_p).all(), "Log probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = log_p.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = torch.distributions.Categorical(logits=log_p).sample()
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = torch.distributions.Categorical(logits=log_p).sample()

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
        log_p, _, _ = self._get_log_p(fixed, state, normalize=normalize)

        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True, consecutive_depot=None, return_debug=False, benchmark_stats=None):
        step_context_start = time.perf_counter() if benchmark_stats is not None else None
        step_context = self._get_parallel_step_context(fixed.node_embeddings, state)
        query = fixed.context_node_projected + self.project_step_context(step_context)
        if benchmark_stats is not None:
            benchmark_stats['seconds']['decode_step_context'] = benchmark_stats['seconds'].get('decode_step_context', 0.0) + (time.perf_counter() - step_context_start)
            benchmark_stats['calls']['decode_step_context'] = benchmark_stats['calls'].get('decode_step_context', 0) + 1

        attention_data_start = time.perf_counter() if benchmark_stats is not None else None
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
        if benchmark_stats is not None:
            benchmark_stats['seconds']['decode_attention_node_data'] = benchmark_stats['seconds'].get('decode_attention_node_data', 0.0) + (time.perf_counter() - attention_data_start)
            benchmark_stats['calls']['decode_attention_node_data'] = benchmark_stats['calls'].get('decode_attention_node_data', 0) + 1

        step_debug = None
        urgency_payload = None
        need_urgency_slack = abs(self.decode_pickup_urgency_bias) > 1e-12
        mask_start = time.perf_counter() if benchmark_stats is not None else None
        aux_outputs = {} if need_urgency_slack else None
        if return_debug:
            mask, step_debug = state.get_mask(
                return_debug=True,
                return_urgency_slack=need_urgency_slack,
                aux_outputs=aux_outputs,
                benchmark_stats=benchmark_stats,
            )
            if need_urgency_slack and isinstance(step_debug, dict):
                urgency_payload = step_debug.get('pickup_tw_slack_hours')
            if urgency_payload is None and need_urgency_slack and isinstance(aux_outputs, dict):
                urgency_payload = aux_outputs.get('pickup_tw_slack_hours')
        else:
            mask = state.get_mask(
                return_urgency_slack=need_urgency_slack,
                aux_outputs=aux_outputs,
                benchmark_stats=benchmark_stats,
            )
            if need_urgency_slack and isinstance(aux_outputs, dict):
                urgency_payload = aux_outputs.get('pickup_tw_slack_hours')
        if benchmark_stats is not None:
            benchmark_stats['seconds']['decode_get_mask'] = benchmark_stats['seconds'].get('decode_get_mask', 0.0) + (time.perf_counter() - mask_start)
            benchmark_stats['calls']['decode_get_mask'] = benchmark_stats['calls'].get('decode_get_mask', 0) + 1
        if consecutive_depot is not None and self.max_consecutive_depot is not None:
            depot_limit_reached = (consecutive_depot >= self.max_consecutive_depot)
            if depot_limit_reached.any():
                state = state._replace(terminal_=state.terminal_ | depot_limit_reached[:, None])
                mask = mask.clone()
                mask[depot_limit_reached, :, :] = True
                mask[depot_limit_reached, :, 0] = False

        all_masked = mask[:, 0, :].all(-1)
        if all_masked.any():
            mask = mask.clone()
            reject_index = state.reject_index
            min_orders_required = int(getattr(state, 'min_orders_per_dispatch', 1))
            in_dispatch = state.prev_a.squeeze(1) != 0
            below_min_dispatch = state.served_orders_since_dispatch.squeeze(1) < min_orders_required
            reject_targets = state._deterministic_reject_order(mask)
            reject_candidate_available = reject_targets >= 0
            prefer_reject_fallback = all_masked & in_dispatch & below_min_dispatch & reject_candidate_available
            if prefer_reject_fallback.any():
                mask[prefer_reject_fallback, :, reject_index] = False

            still_all_masked = mask[:, 0, :].all(-1)
            if still_all_masked.any():
                state = state._replace(terminal_=state.terminal_ | still_all_masked[:, None])
                mask[still_all_masked, :, :] = True
                mask[still_all_masked, :, 0] = False

        if return_debug:
            service_mask = mask[:, 0, 1:1 + 2 * state.n_orders]
            pickup_mask = mask[:, 0, 1:1 + state.n_orders]
            delivery_mask = mask[:, 0, 1 + state.n_orders:1 + 2 * state.n_orders]
            step_debug['diag_steps'] = torch.ones(mask.size(0), device=mask.device)
            step_debug['diag_feasible_service'] = (~service_mask).sum(-1).float()
            step_debug['diag_feasible_pickups'] = (~pickup_mask).sum(-1).float()
            step_debug['diag_feasible_deliveries'] = (~delivery_mask).sum(-1).float()
            step_debug['diag_any_service_feasible'] = (~service_mask).any(-1).float()
            step_debug['diag_depot_only'] = ((~mask[:, 0, 0]) & service_mask.all(-1) & mask[:, 0, -1]).float()
            step_debug['diag_reject_available_rate'] = (~mask[:, 0, -1]).float()

        logits_start = time.perf_counter() if benchmark_stats is not None else None
        log_p = self._one_to_many_logits(query, step_context, glimpse_K, glimpse_V, logit_K, mask)
        urgency_bias = self._build_pickup_urgency_bias(
            state,
            mask,
            urgency_payload,
            dtype=log_p.dtype,
        )
        if urgency_bias is not None:
            pickup_start = 1
            pickup_end = 1 + state.n_orders
            log_p[:, :, pickup_start:pickup_end] = log_p[:, :, pickup_start:pickup_end] + urgency_bias
            if self.mask_logits:
                log_p = log_p.masked_fill(mask, -math.inf)
        if benchmark_stats is not None:
            benchmark_stats['seconds']['decode_logits'] = benchmark_stats['seconds'].get('decode_logits', 0.0) + (time.perf_counter() - logits_start)
            benchmark_stats['calls']['decode_logits'] = benchmark_stats['calls'].get('decode_logits', 0) + 1

        normalize_start = time.perf_counter() if benchmark_stats is not None and normalize else None
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        if benchmark_stats is not None and normalize:
            benchmark_stats['seconds']['decode_log_softmax'] = benchmark_stats['seconds'].get('decode_log_softmax', 0.0) + (time.perf_counter() - normalize_start)
            benchmark_stats['calls']['decode_log_softmax'] = benchmark_stats['calls'].get('decode_log_softmax', 0) + 1

        invalid_value_rows = (~(torch.isfinite(log_p) | torch.isneginf(log_p))).any(dim=-1).squeeze(1)
        if invalid_value_rows.any():
            mask = mask.clone()
            mask[invalid_value_rows, :, :] = True
            mask[invalid_value_rows, :, 0] = False
            log_p = log_p.clone()
            log_p[invalid_value_rows, :, :] = -math.inf
            log_p[invalid_value_rows, :, 0] = 0.0

        return log_p, mask, step_debug

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
                        normalized_time[:, :, None],
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

    def _build_pickup_urgency_bias(self, state, mask, pickup_tw_slack_hours, dtype):
        if abs(self.decode_pickup_urgency_bias) <= 1e-12:
            return None
        n_orders = int(state.n_orders)
        if n_orders <= 0 or pickup_tw_slack_hours is None:
            return None

        if pickup_tw_slack_hours.dim() == 2:
            slack = pickup_tw_slack_hours[:, None, :]
        else:
            slack = pickup_tw_slack_hours

        if slack.size(-1) != n_orders:
            return None

        horizon = max(float(self.decode_pickup_urgency_horizon_hours), 1e-6)
        urgency = torch.clamp(horizon - slack, min=0.0, max=horizon) / horizon

        pickup_start = 1
        pickup_end = 1 + n_orders
        pickup_mask = mask[:, :, pickup_start:pickup_end]
        urgency = torch.where(pickup_mask, torch.zeros_like(urgency), urgency)
        return float(self.decode_pickup_urgency_bias) * urgency.to(dtype)

    def _one_to_many_logits(self, query, step_context, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            node_mask = mask[:, :, :-1].clone()
            all_node_masked = node_mask.all(-1)
            if all_node_masked.any():
                node_mask[all_node_masked] = False
            compatibility[node_mask[:, :, None, :][None, :, :, :, :].expand_as(compatibility)] = -math.inf

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)
        final_Q = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
        )

        node_logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        reject_logit = self.reject_proj(step_context)
        logits = torch.cat((node_logits, reject_logit), dim=-1)

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits

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
