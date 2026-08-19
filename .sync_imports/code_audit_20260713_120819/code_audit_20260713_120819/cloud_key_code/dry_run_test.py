import os
import sys
import types

import torch

# Add current directory to path
sys.path.append(os.getcwd())

from nets.attention_model import AttentionModel, set_decode_type
from problem_mcvrptw_v2 import MCVRPPDTW, MCVRPPDTWDataset, Config
from run_training_optimized import POMOTrainerOptimized
from state_mcvrptw_v2 import StateMCVRPPDTW


def _single_order_state():
    batch_size = 1
    n_orders = 5
    dataset = MCVRPPDTWDataset(num_samples=batch_size, graph_size=n_orders)
    sample = dataset[0]
    return {
        'depot': sample['depot'].unsqueeze(0),
        'loc': sample['loc'].unsqueeze(0),
        'node_type': sample['node_type'].unsqueeze(0),
        'demand_passenger': sample['demand_passenger'].unsqueeze(0),
        'demand_cargo': sample['demand_cargo'].unsqueeze(0),
        'time_windows': sample['time_windows'].unsqueeze(0),
    }, n_orders


def _build_shared_ride_case():
    n_orders = 3
    depot = torch.tensor([[0.20, 0.20]], dtype=torch.float)
    pickup_locs = torch.tensor(
        [[0.25, 0.25], [0.28, 0.26], [0.80, 0.80]], dtype=torch.float
    )
    delivery_locs = torch.tensor(
        [[0.25, 0.35], [0.30, 0.36], [0.82, 0.82]], dtype=torch.float
    )
    loc = torch.cat([pickup_locs, delivery_locs], dim=0).unsqueeze(0)
    node_type = torch.ones(1, 2 * n_orders, dtype=torch.float)
    demand_passenger = torch.zeros(1, 2 * n_orders, dtype=torch.float)
    demand_cargo = torch.zeros(1, 2 * n_orders, dtype=torch.float)
    per_order = 1.0 / Config.PASSENGER_CAPACITY
    demand_passenger[0, :n_orders] = per_order
    demand_passenger[0, n_orders:] = -per_order
    time_windows = torch.tensor([[[10.0, 16.0]] * (2 * n_orders)], dtype=torch.float)
    return {
        'depot': depot,
        'loc': loc,
        'node_type': node_type,
        'demand_passenger': demand_passenger,
        'demand_cargo': demand_cargo,
        'time_windows': time_windows,
    }


def _build_three_order_case():
    n_orders = 4
    depot = torch.tensor([[0.20, 0.20]], dtype=torch.float)
    pickup_locs = torch.tensor(
        [[0.25, 0.25], [0.28, 0.26], [0.31, 0.27], [0.80, 0.80]], dtype=torch.float
    )
    delivery_locs = torch.tensor(
        [[0.25, 0.35], [0.30, 0.36], [0.33, 0.37], [0.82, 0.82]], dtype=torch.float
    )
    loc = torch.cat([pickup_locs, delivery_locs], dim=0).unsqueeze(0)
    node_type = torch.ones(1, 2 * n_orders, dtype=torch.float)
    demand_passenger = torch.zeros(1, 2 * n_orders, dtype=torch.float)
    demand_cargo = torch.zeros(1, 2 * n_orders, dtype=torch.float)
    per_order = 1.0 / Config.PASSENGER_CAPACITY
    demand_passenger[0, :n_orders] = per_order
    demand_passenger[0, n_orders:] = -per_order
    time_windows = torch.tensor([[[10.0, 16.0]] * (2 * n_orders)], dtype=torch.float)
    return {
        'depot': depot,
        'loc': loc,
        'node_type': node_type,
        'demand_passenger': demand_passenger,
        'demand_cargo': demand_cargo,
        'time_windows': time_windows,
    }


def _collate_dataset(dataset, batch_size):
    sample0 = dataset[0]
    return {
        key: torch.stack([dataset[i][key] for i in range(batch_size)], dim=0)
        if torch.is_tensor(sample0[key]) else sample0[key]
        for key in sample0.keys()
    }


def _repeat_for_pomo(batch, pomo_size):
    return {
        key: value.repeat_interleave(pomo_size, dim=0) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _reshape_pomo_tensor(value, base_batch_size, pomo_size):
    return value.reshape(base_batch_size, pomo_size, *value.shape[1:])


def _build_test_attention_model(shrink_size, decode_pickup_urgency_bias=0.0, decode_pickup_urgency_horizon_hours=1.0):
    model = AttentionModel(
        embedding_dim=64,
        hidden_dim=64,
        problem=MCVRPPDTW,
        n_encode_layers=2,
        n_heads=8,
        tanh_clipping=10.0,
        normalization='batch',
        shrink_size=shrink_size,
        reject_init_bias=-2.5,
        decode_pickup_urgency_bias=decode_pickup_urgency_bias,
        decode_pickup_urgency_horizon_hours=decode_pickup_urgency_horizon_hours,
    )
    model.eval()
    set_decode_type(model, 'greedy')
    return model


def basic_dry_run():
    print('=' * 60)
    print('VRP 重构逻辑 Dry-run 验证')
    print('=' * 60)

    input_data, n_orders = _single_order_state()
    state = StateMCVRPPDTW.initialize(input_data)
    print(f'初始状态: 时间={state.current_time.item():.2f}, 车辆={state.used_vehicles.item()}')

    mask = state.get_mask()
    last_node = 2 * n_orders
    print(f'初始掩码: 节点0(换车)={mask[0, 0, 0].item()}, 节点{last_node}(最后一个delivery)={mask[0, 0, last_node].item()}')
    print('预期: 初始时由于没有任务完成且有任务可做，回库点应被屏蔽。')

    pickup_idx = 1
    state = state.update(torch.tensor([pickup_idx]))
    print(f'\n步骤1 (访问Pickup {pickup_idx}): 时间={state.current_time.item():.2f}, 乘客容量={state.used_capacity_passenger.item():.2f}')

    mask = state.get_mask()
    print(f'带货掩码: 节点0(换车)={mask[0, 0, 0].item()}, 节点{last_node}(最后一个delivery)={mask[0, 0, last_node].item()}')
    print('预期: 带货时回库点必须被屏蔽 (True).')

    delivery_idx = pickup_idx + n_orders
    state = state.update(torch.tensor([delivery_idx]))
    print(f'\n步骤2 (访问Delivery {delivery_idx}): 时间={state.current_time.item():.2f}, 乘客容量={state.used_capacity_passenger.item():.2f}')

    mask = state.get_mask()
    print(f'空载掩码: 节点0(换车)={mask[0, 0, 0].item()}, 节点{last_node}(最后一个delivery)={mask[0, 0, last_node].item()}')

    state = state.update(torch.tensor([0]))
    print(f'\n步骤3 (返回车场-换新车): 时间={state.current_time.item():.2f}, 车辆={state.used_vehicles.item()}')
    print('预期: 时间应重置为10.0，车辆数应增加 1（当前实现记已启用车辆数）。')

    pi = torch.tensor([[0, 1, 1 + n_orders, 0, 2, 2 + n_orders, 0]])
    costs, details = MCVRPPDTW.get_costs(input_data, pi, return_details=True)

    print('\n最终成本验证:')
    print(f'  总成本: {costs.item():.2f} 元')
    print(f"  车辆成本: {details['vehicle_cost'].item():.2f} 元")
    print(f"  使用车辆数: {details['used_vehicles'].item()}")
    print(f"  能耗成本: {details['energy_cost'].item():.2f} 元")
    print(f"  时间窗超时惩罚: {details['trip_overtime_penalty'].item():.2f} 元")


def shared_mask_regression():
    print('\n' + '=' * 60)
    print('共享 mask 回归测试')
    print('=' * 60)

    input_data = _build_shared_ride_case()
    stable_state = StateMCVRPPDTW.initialize(input_data)
    stable_state = stable_state.update(torch.tensor([1]))
    stable_mask, stable_debug = stable_state.get_mask(return_debug=True)
    stable_pickup2_masked = bool(stable_mask[0, 0, 2].item())
    print(f'稳定模式: pickup2 masked={stable_pickup2_masked}')
    assert stable_pickup2_masked is True, 'cap=1 稳定模式必须保持第二个 pickup 被屏蔽'

    shared_state = StateMCVRPPDTW.initialize(
        input_data,
        max_concurrent_open_orders=6,
        enable_delivery_viability=True,
        enable_viability_fallback=True,
    )
    shared_state = shared_state.update(torch.tensor([1]))
    shared_mask, shared_debug = shared_state.get_mask(return_debug=True)
    pickup2_masked = bool(shared_mask[0, 0, 2].item())
    pickup3_masked = bool(shared_mask[0, 0, 3].item())
    print(f'共享模式: pickup2 masked={pickup2_masked}, pickup3 masked={pickup3_masked}')
    print(f"Open count={shared_debug['diag_open_started_count'].item():.2f}, second pickup feasible={shared_debug['diag_second_pickup_feasible'].item():.2f}")
    assert pickup2_masked is False, '共享可行的第二个 pickup 仍被错误屏蔽'
    assert pickup3_masked is True, '不可闭环的远端 pickup 未被屏蔽'
    assert shared_debug['diag_reject_inroute_available'].item() == 0.0, 'in-route reject 不应重新出现'

    shared_state = shared_state.update(torch.tensor([2]))
    after_second_mask, after_second_debug = shared_state.get_mask(return_debug=True)
    print(f"两单并发后 open count={after_second_debug['diag_open_started_count'].item():.2f}")
    assert after_second_debug['diag_open_started_count'].item() >= 2.0, '共享状态下应至少存在 2 个 open 单'

    # 第三个近场 pickup 仍应可行，因为目标语义是无硬并发上限
    third_pickup_masked = bool(after_second_mask[0, 0, 3].item())
    print(f'两单并发后 pickup3 masked={third_pickup_masked}')

    shared_route = torch.tensor([[0, 1, 2, 4, 5, 0]])
    _, details = MCVRPPDTW.get_costs(input_data, shared_route, return_details=True)
    print(f"Pickup-only={details['pickup_only_orders'].item():.2f}, Started-not-completed={details['started_not_completed_orders'].item():.2f}")
    assert details['pickup_only_orders'].item() == 0.0, '共享测试路线不应出现 pickup-only'
    assert details['started_not_completed_orders'].item() == 0.0, '共享测试路线不应出现 started-not-completed'


def viability_fallback_regression():
    print('\n' + '=' * 60)
    print('delivery viability fallback 测试')
    print('=' * 60)

    input_data = _build_three_order_case()
    state = StateMCVRPPDTW.initialize(
        input_data,
        max_concurrent_open_orders=6,
        enable_delivery_viability=True,
        enable_viability_fallback=True,
    )
    state = state.update(torch.tensor([1]))
    state = state.update(torch.tensor([2]))
    mask, debug = state.get_mask(return_debug=True)
    print(f"delivery viability masked={debug['diag_delivery_viability_masked'].item():.2f}, fallback={debug['diag_delivery_viability_fallback'].item():.2f}")
    assert debug['diag_reject_inroute_available'].item() == 0.0, '共享状态中 reject 仍不可用'
    feasible_deliveries = (~mask[0, 0, 5:9]).sum().item()
    print(f'当前可行 delivery 数={feasible_deliveries}')
    assert feasible_deliveries >= 1, 'delivery viability 不应把所有出口都锁死'


def pickup_time_update_regression():
    print('\n' + '=' * 60)
    print('passenger_pickup_time update 回归测试')
    print('=' * 60)

    input_data = _build_shared_ride_case()
    base_state = StateMCVRPPDTW.initialize(
        input_data,
        max_concurrent_open_orders=6,
        enable_delivery_viability=True,
        enable_viability_fallback=True,
    )

    first_state = base_state.update(torch.tensor([1]))
    first_times = first_state.passenger_pickup_time.squeeze(1)[0]
    expected_first = torch.full_like(first_times, -1.0)
    expected_first[0] = first_state.current_time.item()
    print(f'after pickup1 times={first_times.tolist()}')
    assert torch.allclose(first_times, expected_first), '第一次 passenger pickup 应只写入第一个订单时间'

    second_state = first_state.update(torch.tensor([2]))
    second_times = second_state.passenger_pickup_time.squeeze(1)[0]
    expected_second = expected_first.clone()
    expected_second[1] = second_state.current_time.item()
    print(f'after pickup2 times={second_times.tolist()}')
    assert torch.allclose(second_times, expected_second), '第二次 passenger pickup 不应覆盖第一个订单时间'

    delivery_state = second_state.update(torch.tensor([4]))
    delivery_times = delivery_state.passenger_pickup_time.squeeze(1)[0]
    assert torch.allclose(delivery_times, expected_second), 'delivery 动作不应重写 passenger pickup time'



def pickup_commitment_next_delivery_equivalence_regression():
    print('\n' + '=' * 60)
    print('pickup_commitment next-delivery 等价测试')
    print('=' * 60)

    input_data = _build_shared_ride_case()
    state = StateMCVRPPDTW.initialize(
        input_data,
        max_concurrent_open_orders=6,
        enable_delivery_viability=True,
        enable_viability_fallback=True,
    )
    state = state.update(torch.tensor([1]))

    mask = state.get_mask(skip_pickup_commitment=True)
    n_orders = state.n_orders
    ids_flat, coords_active, node_type_active, time_windows_active, _, _ = state._active_views()
    arrival_time = state.current_time + (coords_active - state.cur_coord).norm(p=2, dim=-1) * state.AREA_SIZE / state.VEHICLE_SPEED
    pickup_coords = coords_active[:, 1:n_orders + 1, :]
    delivery_coords = coords_active[:, n_orders + 1:2 * n_orders + 1, :]
    delivery_earliest = time_windows_active[:, n_orders + 1:2 * n_orders + 1, 0]
    pickup_finish = torch.maximum(
        arrival_time[:, 1:n_orders + 1],
        time_windows_active[:, 1:n_orders + 1, 0]
    ) + state.SERVICE_TIME
    delivery_to_depot_time = (
        (delivery_coords - coords_active[:, 0:1, :]).norm(p=2, dim=-1) * state.AREA_SIZE / state.VEHICLE_SPEED
    )
    trip_start_after_pickup = torch.where(state.prev_a == 0, state.current_time, state.trip_start_time).squeeze(1)
    passenger_orders = (node_type_active[:, 1:n_orders + 1] == 1)
    passenger_pickup_times_before = state.passenger_pickup_time.squeeze(1)
    direct_ride_time = (
        (pickup_coords - delivery_coords).norm(p=2, dim=-1) * state.AREA_SIZE / state.VEHICLE_SPEED
    )
    open_before = state.get_open_started_mask()[0]
    open_before_bits = state._open_mask_to_bits(open_before)
    candidate_indices = torch.nonzero((~mask[0, 0, 1:n_orders + 1]), as_tuple=False).squeeze(-1).tolist()
    assert candidate_indices, '测试状态下应存在可选 pickup 候选'

    for candidate_slot in candidate_indices:
        candidate_idx = int(candidate_slot)
        open_after = open_before.clone()
        open_after[candidate_idx] = True
        open_after_bits = open_before_bits | (1 << candidate_idx)
        passenger_pickup_times_after = passenger_pickup_times_before[0].clone()
        if bool(passenger_orders[0, candidate_idx].item()):
            passenger_pickup_times_after[candidate_idx] = pickup_finish[0, candidate_idx]

        start_node_key = ('pickup', candidate_idx)
        helper_has_delivery = state._has_post_pickup_next_delivery(
            pickup_coords[0, candidate_idx],
            pickup_finish[0, candidate_idx],
            trip_start_after_pickup[0],
            open_after,
            delivery_coords[0],
            delivery_earliest[0],
            delivery_to_depot_time[0],
            passenger_orders[0],
            passenger_pickup_times_after,
            direct_ride_time[0],
            open_bits=open_after_bits,
            start_node=start_node_key,
        )

        helper_completion = state._has_feasible_open_completion(
            pickup_coords[0, candidate_idx],
            pickup_finish[0, candidate_idx],
            trip_start_after_pickup[0],
            open_after,
            delivery_coords[0],
            delivery_earliest[0],
            delivery_to_depot_time[0],
            passenger_orders[0],
            passenger_pickup_times_after,
            direct_ride_time[0],
            open_bits=open_after_bits,
            start_node=start_node_key,
        )
        helper_completion_reason = state._has_feasible_open_completion(
            pickup_coords[0, candidate_idx],
            pickup_finish[0, candidate_idx],
            trip_start_after_pickup[0],
            open_after,
            delivery_coords[0],
            delivery_earliest[0],
            delivery_to_depot_time[0],
            passenger_orders[0],
            passenger_pickup_times_after,
            direct_ride_time[0],
            return_reason=True,
            open_bits=open_after_bits,
            start_node=start_node_key,
        )
        assert helper_completion == helper_completion_reason[0], 'return_reason 不应改变 completion 布尔结果'

        candidate_node = candidate_idx + 1
        next_state = state.update(torch.tensor([candidate_node]), current_mask=mask)
        next_state_times = next_state.passenger_pickup_time.squeeze(1)[0]
        assert torch.allclose(next_state_times, passenger_pickup_times_after), (
            f'pickup {candidate_node} 后 next_state.passenger_pickup_time 与手工构造不一致'
        )
        next_mask = next_state.get_mask(skip_pickup_commitment=True)
        next_feasible_deliveries = int((~next_mask[0, 0, n_orders + 1:2 * n_orders + 1]).sum().item())
        full_mask_has_delivery = next_feasible_deliveries > 0
        print(
            f'candidate pickup={candidate_node} helper_has_delivery={helper_has_delivery} '
            f'full_mask_has_delivery={full_mask_has_delivery} completion={helper_completion} '
            f'reason={helper_completion_reason[1]}'
        )
        assert helper_has_delivery == full_mask_has_delivery, (
            f'pickup {candidate_node} 的 next-delivery helper 与 full next_mask 不一致'
        )


def attention_shrink_pomo_regression():
    print('\n' + '=' * 60)
    print('attention shrink + POMO 回归测试')
    print('=' * 60)

    base_batch_size = 16
    pomo_size = 2
    dataset = MCVRPPDTWDataset(num_samples=base_batch_size, graph_size=8, seed=2026)
    batch = _collate_dataset(dataset, base_batch_size)
    repeated_batch = _repeat_for_pomo(batch, pomo_size)
    state_kwargs = {
        'max_concurrent_open_orders': 6,
        'enable_delivery_viability': True,
        'enable_viability_fallback': False,
    }

    torch.manual_seed(1234)
    model_no_shrink = _build_test_attention_model(shrink_size=None)
    model_shrink = _build_test_attention_model(shrink_size=4)
    model_shrink.load_state_dict(model_no_shrink.state_dict())

    with torch.no_grad():
        cost_no_shrink, ll_no_shrink, pi_no_shrink, debug_no_shrink = model_no_shrink(
            batch,
            return_pi=True,
            state_kwargs=state_kwargs,
            return_debug=True,
            logical_pomo_size=pomo_size,
        )
        cost_shrink, ll_shrink, pi_shrink, debug_shrink = model_shrink(
            batch,
            return_pi=True,
            state_kwargs=state_kwargs,
            return_debug=True,
            logical_pomo_size=pomo_size,
        )
        cost_ref, ll_ref, pi_ref, debug_ref = model_no_shrink(
            repeated_batch,
            return_pi=True,
            state_kwargs=state_kwargs,
            return_debug=True,
        )

    cost_no_shrink_flat = cost_no_shrink
    ll_no_shrink_flat = ll_no_shrink
    pi_no_shrink_flat = pi_no_shrink
    cost_shrink_flat = cost_shrink
    ll_shrink_flat = ll_shrink
    pi_shrink_flat = pi_shrink

    cost_no_shrink = _reshape_pomo_tensor(cost_no_shrink_flat, base_batch_size, pomo_size)
    ll_no_shrink = _reshape_pomo_tensor(ll_no_shrink_flat, base_batch_size, pomo_size)
    pi_no_shrink = _reshape_pomo_tensor(pi_no_shrink_flat, base_batch_size, pomo_size)
    cost_shrink = _reshape_pomo_tensor(cost_shrink_flat, base_batch_size, pomo_size)
    ll_shrink = _reshape_pomo_tensor(ll_shrink_flat, base_batch_size, pomo_size)
    pi_shrink = _reshape_pomo_tensor(pi_shrink_flat, base_batch_size, pomo_size)
    cost_ref = _reshape_pomo_tensor(cost_ref, base_batch_size, pomo_size)
    ll_ref = _reshape_pomo_tensor(ll_ref, base_batch_size, pomo_size)
    pi_ref = _reshape_pomo_tensor(pi_ref, base_batch_size, pomo_size)

    print(f'no_shrink pi shape={tuple(pi_no_shrink.shape)}, shrink pi shape={tuple(pi_shrink.shape)}')
    assert pi_no_shrink.shape == pi_shrink.shape, 'shrink/no-shrink 的 pi shape 不一致'
    assert torch.equal(pi_no_shrink, pi_shrink), 'shrink/no-shrink greedy decode 序列不一致'
    assert torch.allclose(cost_no_shrink, cost_shrink), 'shrink/no-shrink cost 不一致'
    assert torch.allclose(ll_no_shrink, ll_shrink), 'shrink/no-shrink log likelihood 不一致'
    assert torch.equal(pi_no_shrink, pi_ref), 'logical POMO 与物理 repeat POMO greedy decode 序列不一致'
    assert torch.allclose(cost_no_shrink, cost_ref), 'logical POMO 与物理 repeat POMO cost 不一致'
    assert torch.allclose(ll_no_shrink, ll_ref), 'logical POMO 与物理 repeat POMO log likelihood 不一致'

    debug_keys = [
        'diag_steps',
        'diag_any_service_feasible',
        'diag_selected_depot',
        'diag_selected_pickup',
        'diag_selected_delivery',
        'diag_selected_reject',
        'diag_mask_pickup_commitment',
        'diag_delivery_viability_masked',
    ]
    for key in debug_keys:
        assert key in debug_no_shrink and key in debug_shrink and key in debug_ref, f'缺少 shrink/logical POMO debug key: {key}'
        assert debug_no_shrink[key].shape == debug_shrink[key].shape == debug_ref[key].shape, f'debug[{key}] shape 不一致'
        assert torch.allclose(debug_no_shrink[key], debug_ref[key]), f'logical POMO debug[{key}] 与物理 repeat POMO 不一致'
        assert torch.isfinite(debug_no_shrink[key]).all(), f'no-shrink debug[{key}] 出现非有限值'
        assert torch.isfinite(debug_shrink[key]).all(), f'shrink debug[{key}] 出现非有限值'


def _build_pickup_urgency_case(second_pickup_missed=False):
    n_orders = 2
    depot = torch.tensor([[0.10, 0.10]], dtype=torch.float)
    pickup_locs = torch.tensor([[0.12, 0.10], [0.11, 0.10]], dtype=torch.float)
    delivery_locs = torch.tensor([[0.12, 0.14], [0.11, 0.14]], dtype=torch.float)
    loc = torch.cat([pickup_locs, delivery_locs], dim=0).unsqueeze(0)

    node_type = torch.ones(1, 2 * n_orders, dtype=torch.float)
    demand_passenger = torch.zeros(1, 2 * n_orders, dtype=torch.float)
    demand_cargo = torch.zeros(1, 2 * n_orders, dtype=torch.float)
    per_order = 1.0 / Config.PASSENGER_CAPACITY
    demand_passenger[0, :n_orders] = per_order
    demand_passenger[0, n_orders:] = -per_order

    pickup2_end = 10.002 if second_pickup_missed else 10.06
    time_windows = torch.tensor([[
        [10.0, 12.0],
        [10.0, pickup2_end],
        [10.0, 16.0],
        [10.0, 16.0],
    ]], dtype=torch.float)

    return {
        'depot': depot,
        'loc': loc,
        'node_type': node_type,
        'demand_passenger': demand_passenger,
        'demand_cargo': demand_cargo,
        'time_windows': time_windows,
    }


def _prepare_fixed_and_state(model, batch):
    with torch.no_grad():
        init_embed = model._init_embed(batch)
        pd_pair_mask = model._build_pd_pair_mask(batch)
        embeddings, _ = model.embedder(init_embed, pd_pair_mask=pd_pair_mask)
        fixed = model._precompute(embeddings)
    state = StateMCVRPPDTW.initialize(
        batch,
        min_orders_per_dispatch=1,
        max_concurrent_open_orders=6,
        enable_delivery_viability=True,
        enable_viability_fallback=False,
    )
    return fixed, state


def _attach_flat_logits(model):
    def _flat_logits(self, query, step_context, glimpse_K, glimpse_V, logit_K, mask):
        logits = torch.zeros(mask.size(), device=mask.device, dtype=query.dtype)
        logits = logits.masked_fill(mask, -float('inf'))
        return logits

    model._one_to_many_logits = types.MethodType(_flat_logits, model)


def urgency_bias_changes_greedy_choice_regression():
    print('\n' + '=' * 60)
    print('pickup urgency bias 改变 greedy 选择回归测试')
    print('=' * 60)

    batch = _build_pickup_urgency_case(second_pickup_missed=False)

    model_base = _build_test_attention_model(shrink_size=None, decode_pickup_urgency_bias=0.0, decode_pickup_urgency_horizon_hours=1.0)
    model_urgency = _build_test_attention_model(shrink_size=None, decode_pickup_urgency_bias=3.0, decode_pickup_urgency_horizon_hours=1.0)
    model_urgency.load_state_dict(model_base.state_dict())
    _attach_flat_logits(model_base)
    _attach_flat_logits(model_urgency)

    fixed_base, state = _prepare_fixed_and_state(model_base, batch)
    fixed_urgency, _ = _prepare_fixed_and_state(model_urgency, batch)

    with torch.no_grad():
        log_p_base, mask_base, _ = model_base._get_log_p(fixed_base, state, normalize=True, return_debug=False)
        log_p_urgency, mask_urgency, _ = model_urgency._get_log_p(fixed_urgency, state, normalize=True, return_debug=False)
        selected_base = model_base._select_node(log_p_base[:, 0, :], mask_base[:, 0, :])
        selected_urgency = model_urgency._select_node(log_p_urgency[:, 0, :], mask_urgency[:, 0, :])

    print(f'base selected={int(selected_base.item())}, urgency selected={int(selected_urgency.item())}')
    assert bool(mask_base[0, 0, 1].item()) is False and bool(mask_base[0, 0, 2].item()) is False, '测试前提失败：两个 pickup 应都可行'
    assert int(selected_base.item()) == 1, '无 urgency 时应按平分 tie-break 选第一个 pickup'
    assert int(selected_urgency.item()) == 2, '开启 urgency 后应优先更紧迫的 pickup'


def urgency_bias_preserves_mask_semantics_regression():
    print('\n' + '=' * 60)
    print('pickup urgency bias 保持 mask 语义回归测试')
    print('=' * 60)

    batch = _build_pickup_urgency_case(second_pickup_missed=True)
    model_urgency = _build_test_attention_model(shrink_size=None, decode_pickup_urgency_bias=50.0, decode_pickup_urgency_horizon_hours=1.0)
    _attach_flat_logits(model_urgency)
    fixed, state = _prepare_fixed_and_state(model_urgency, batch)

    with torch.no_grad():
        log_p, mask, _ = model_urgency._get_log_p(fixed, state, normalize=False, return_debug=False)
        selected = model_urgency._select_node(log_p[:, 0, :], mask[:, 0, :])

    print(f'masked pickup2={bool(mask[0,0,2].item())}, selected={int(selected.item())}')
    assert bool(mask[0, 0, 2].item()) is True, '测试前提失败：pickup2 应被 pickup TW 硬屏蔽'
    assert torch.isneginf(log_p[0, 0, 2]), '被 mask 的 pickup2 logit 必须保持 -inf'
    assert int(selected.item()) != 2, '被 mask 的 pickup2 不可被选中'


def urgency_bias_training_path_slack_available_regression():
    print('\n' + '=' * 60)
    print('pickup urgency bias 训练路径 slack 可用性回归测试')
    print('=' * 60)

    batch = _build_pickup_urgency_case(second_pickup_missed=False)

    model_base = _build_test_attention_model(shrink_size=None, decode_pickup_urgency_bias=0.0, decode_pickup_urgency_horizon_hours=1.0)
    model_urgency = _build_test_attention_model(shrink_size=None, decode_pickup_urgency_bias=3.0, decode_pickup_urgency_horizon_hours=1.0)
    model_urgency.load_state_dict(model_base.state_dict())
    _attach_flat_logits(model_base)
    _attach_flat_logits(model_urgency)

    fixed_base, state_base = _prepare_fixed_and_state(model_base, batch)
    fixed_urgency, state_urgency = _prepare_fixed_and_state(model_urgency, batch)

    observed = {'training_calls': 0, 'slack_populated': 0}
    original_get_mask = StateMCVRPPDTW.get_mask

    def _wrapped_get_mask(self, *args, **kwargs):
        aux_outputs = kwargs.get('aux_outputs')
        return_debug = bool(kwargs.get('return_debug', False))
        return_urgency_slack = bool(kwargs.get('return_urgency_slack', False))
        result = original_get_mask(self, *args, **kwargs)
        if (not return_debug) and return_urgency_slack:
            observed['training_calls'] += 1
            if isinstance(aux_outputs, dict) and ('pickup_tw_slack_hours' in aux_outputs):
                observed['slack_populated'] += 1
        return result

    StateMCVRPPDTW.get_mask = _wrapped_get_mask
    try:
        with torch.no_grad():
            log_p_base, mask_base, _ = model_base._get_log_p(fixed_base, state_base, normalize=True, return_debug=False)
            log_p_urgency, mask_urgency, _ = model_urgency._get_log_p(fixed_urgency, state_urgency, normalize=True, return_debug=False)
            selected_base = model_base._select_node(log_p_base[:, 0, :], mask_base[:, 0, :])
            selected_urgency = model_urgency._select_node(log_p_urgency[:, 0, :], mask_urgency[:, 0, :])
    finally:
        StateMCVRPPDTW.get_mask = original_get_mask

    print(f"training_path_calls={observed['training_calls']}, slack_populated={observed['slack_populated']}, base={int(selected_base.item())}, urgency={int(selected_urgency.item())}")
    assert observed['training_calls'] > 0, 'return_debug=False 训练路径应请求 urgency slack'
    assert observed['slack_populated'] > 0, '训练路径 aux_outputs 中应写入 pickup_tw_slack_hours'
    assert int(selected_base.item()) == 1 and int(selected_urgency.item()) == 2, '训练路径下 urgency 应实际影响选择'


def pomo_baseline_mode_regression():
    print('\n' + '=' * 60)
    print('POMO baseline mode 回归测试')
    print('=' * 60)

    costs_single = torch.tensor([[1.0], [3.0]], dtype=torch.float)
    log_probs_single = torch.tensor([[0.2], [0.4]], dtype=torch.float)
    loss_auto_single, min_cost_single = POMOTrainerOptimized._pomo_loss(costs_single, log_probs_single, baseline_mode='auto')
    loss_batch_single, min_cost_batch_single = POMOTrainerOptimized._pomo_loss(costs_single, log_probs_single, baseline_mode='batch_mean')
    assert torch.allclose(loss_auto_single, loss_batch_single), 'pomo=1 时 auto 应等价于 batch_mean'
    assert torch.allclose(min_cost_single, min_cost_batch_single), 'min_cost 不应受 baseline_mode 影响'

    costs_multi = torch.tensor([[1.0, 3.0], [5.0, 9.0]], dtype=torch.float)
    log_probs_multi = torch.tensor([[0.2, 0.4], [0.6, 0.8]], dtype=torch.float)
    loss_auto_multi, min_cost_auto_multi = POMOTrainerOptimized._pomo_loss(costs_multi, log_probs_multi, baseline_mode='auto')
    loss_instance_multi, min_cost_instance_multi = POMOTrainerOptimized._pomo_loss(costs_multi, log_probs_multi, baseline_mode='instance_mean')
    assert torch.allclose(loss_auto_multi, loss_instance_multi), 'pomo>1 时 auto 应等价于 instance_mean'
    assert torch.allclose(min_cost_auto_multi, min_cost_instance_multi), 'min_cost 不应受 baseline_mode 影响'

    loss_batch_multi, _ = POMOTrainerOptimized._pomo_loss(costs_multi, log_probs_multi, baseline_mode='batch_mean')
    assert not torch.allclose(loss_batch_multi, loss_instance_multi), 'batch_mean 与 instance_mean 在多 POMO 下应可区分'

    try:
        POMOTrainerOptimized._pomo_loss(costs_single, log_probs_single, baseline_mode='instance_mean')
    except ValueError as exc:
        assert 'requires pomo_size > 1' in str(exc)
    else:
        raise AssertionError('pomo=1 + instance_mean 应显式报错，避免静默退化')


def business_priority_regression():
    print('\n' + '=' * 60)
    print('Business priority checkpoint 回归测试')
    print('=' * 60)

    clean = {
        'service_rate': 0.80,
        'avg_objective': 1500.0,
        'avg_unfulfilled_orders': 0.0,
        'avg_pickup_only_orders': 0.0,
        'avg_started_not_completed_orders': 0.0,
        'avg_untouched_unrejected_orders': 0.0,
    }
    dirty_higher_service = {
        'service_rate': 0.90,
        'avg_objective': 1200.0,
        'avg_unfulfilled_orders': 0.1,
        'avg_pickup_only_orders': 0.0,
        'avg_started_not_completed_orders': 0.0,
        'avg_untouched_unrejected_orders': 0.0,
    }
    cleaner_better_service = {
        'service_rate': 0.82,
        'avg_objective': 1600.0,
        'avg_unfulfilled_orders': 0.0,
        'avg_pickup_only_orders': 0.0,
        'avg_started_not_completed_orders': 0.0,
        'avg_untouched_unrejected_orders': 0.0,
    }
    cleaner_better_objective = {
        'service_rate': 0.82,
        'avg_objective': 1400.0,
        'avg_unfulfilled_orders': 0.0,
        'avg_pickup_only_orders': 0.0,
        'avg_started_not_completed_orders': 0.0,
        'avg_untouched_unrejected_orders': 0.0,
    }

    assert POMOTrainerOptimized._is_business_clean(clean) is True, '全零业务桶应视为 clean'
    assert POMOTrainerOptimized._is_business_clean(dirty_higher_service) is False, '存在 silent-miss 桶时不应视为 clean'
    assert POMOTrainerOptimized._business_priority_key(clean) > POMOTrainerOptimized._business_priority_key(dirty_higher_service), (
        'clean checkpoint 应优先于更高 service 但不 clean 的 checkpoint'
    )
    assert POMOTrainerOptimized._business_priority_key(cleaner_better_service) > POMOTrainerOptimized._business_priority_key(clean), (
        '在 clean 集合内应优先选择更高 service_rate'
    )
    assert POMOTrainerOptimized._business_priority_key(cleaner_better_objective) > POMOTrainerOptimized._business_priority_key(cleaner_better_service), (
        '在 clean 且 service 相同的情况下应优先选择更低 objective'
    )


def cargo_pickup_hard_timewindow_regression():
    print('\n' + '=' * 60)
    print('cargo pickup 硬时间窗回归测试')
    print('=' * 60)

    old_hard = Config.HARD_CARGO_PICKUP_TIMEWINDOW
    Config.HARD_CARGO_PICKUP_TIMEWINDOW = True
    try:
        depot = torch.tensor([[0.2, 0.2]], dtype=torch.float)
        pickup = torch.tensor([[0.8, 0.8]], dtype=torch.float)
        delivery = torch.tensor([[0.81, 0.81]], dtype=torch.float)
        loc = torch.cat([pickup, delivery], dim=0).unsqueeze(0)
        node_type = torch.zeros(1, 2, dtype=torch.float)
        demand_passenger = torch.zeros(1, 2, dtype=torch.float)
        demand_cargo = torch.tensor([[0.2, -0.2]], dtype=torch.float)
        time_windows = torch.tensor([[[10.0, 10.05], [10.0, 16.0]]], dtype=torch.float)
        batch = {
            'depot': depot,
            'loc': loc,
            'node_type': node_type,
            'demand_passenger': demand_passenger,
            'demand_cargo': demand_cargo,
            'time_windows': time_windows,
        }
        state = StateMCVRPPDTW.initialize(batch, min_orders_per_dispatch=1)
        mask = state.get_mask()
        pickup_masked = bool(mask[0, 0, 1].item())
        print(f'cargo pickup masked={pickup_masked}')
        assert pickup_masked is True, 'cargo pickup 超窗时必须被硬屏蔽'
    finally:
        Config.HARD_CARGO_PICKUP_TIMEWINDOW = old_hard


def cargo_piecewise_delay_cost_regression():
    print('\n' + '=' * 60)
    print('cargo delivery 分段迟到成本回归测试')
    print('=' * 60)

    old_cfg = (
        Config.CARGO_DELAY_COST,
        Config.CARGO_DELAY_TIER1_MIN,
        Config.CARGO_DELAY_TIER2_MIN,
        Config.CARGO_DELAY_COST_30_60,
        Config.CARGO_DELAY_COST_60_PLUS,
    )
    Config.CARGO_DELAY_COST = 0.06
    Config.CARGO_DELAY_TIER1_MIN = 30.0
    Config.CARGO_DELAY_TIER2_MIN = 60.0
    Config.CARGO_DELAY_COST_30_60 = 0.18
    Config.CARGO_DELAY_COST_60_PLUS = 0.36
    try:
        depot = torch.tensor([[0.2, 0.2]], dtype=torch.float)
        pickup = torch.tensor([[0.2, 0.2]], dtype=torch.float)
        delivery = torch.tensor([[0.8, 0.8]], dtype=torch.float)
        loc = torch.cat([pickup, delivery], dim=0).unsqueeze(0)
        node_type = torch.zeros(1, 2, dtype=torch.float)
        demand_passenger = torch.zeros(1, 2, dtype=torch.float)
        demand_cargo = torch.tensor([[0.2, -0.2]], dtype=torch.float)
        time_windows = torch.tensor([[[10.0, 16.0], [10.0, 11.0]]], dtype=torch.float)
        batch = {
            'depot': depot,
            'loc': loc,
            'node_type': node_type,
            'demand_passenger': demand_passenger,
            'demand_cargo': demand_cargo,
            'time_windows': time_windows,
        }
        pi = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
        _, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)
        late_min = float(details['cargo_delay_minutes'].item())
        tier1 = min(late_min, Config.CARGO_DELAY_TIER1_MIN)
        tier2 = min(max(late_min - Config.CARGO_DELAY_TIER1_MIN, 0.0), Config.CARGO_DELAY_TIER2_MIN - Config.CARGO_DELAY_TIER1_MIN)
        tier3 = max(late_min - Config.CARGO_DELAY_TIER2_MIN, 0.0)
        expected = (
            tier1 * Config.CARGO_DELAY_COST
            + tier2 * Config.CARGO_DELAY_COST_30_60
            + tier3 * Config.CARGO_DELAY_COST_60_PLUS
        )
        got = float(details['cargo_delay_cost_raw'].item())
        print(f'late_min={late_min:.2f}, expected={expected:.6f}, got={got:.6f}')
        assert abs(got - expected) < 1e-5, 'cargo 分段迟到成本与手算不一致'
    finally:
        (
            Config.CARGO_DELAY_COST,
            Config.CARGO_DELAY_TIER1_MIN,
            Config.CARGO_DELAY_TIER2_MIN,
            Config.CARGO_DELAY_COST_30_60,
            Config.CARGO_DELAY_COST_60_PLUS,
        ) = old_cfg


def min_orders_per_dispatch_regression():
    print('\n' + '=' * 60)
    print('发车最少完成订单数硬约束回归测试')
    print('=' * 60)

    n_orders = 4
    depot = torch.tensor([[0.2, 0.2]], dtype=torch.float)
    pickup = torch.tensor([[0.24, 0.24], [0.26, 0.24], [0.28, 0.24], [0.30, 0.24]], dtype=torch.float)
    delivery = torch.tensor([[0.24, 0.32], [0.26, 0.32], [0.28, 0.32], [0.30, 0.32]], dtype=torch.float)
    loc = torch.cat([pickup, delivery], dim=0).unsqueeze(0)
    node_type = torch.ones(1, 2 * n_orders, dtype=torch.float)
    demand_passenger = torch.zeros(1, 2 * n_orders, dtype=torch.float)
    per = 1.0 / Config.PASSENGER_CAPACITY
    demand_passenger[0, :n_orders] = per
    demand_passenger[0, n_orders:] = -per
    demand_cargo = torch.zeros(1, 2 * n_orders, dtype=torch.float)
    time_windows = torch.tensor([[[10.0, 16.0]] * (2 * n_orders)], dtype=torch.float)
    batch = {
        'depot': depot,
        'loc': loc,
        'node_type': node_type,
        'demand_passenger': demand_passenger,
        'demand_cargo': demand_cargo,
        'time_windows': time_windows,
    }

    state = StateMCVRPPDTW.initialize(batch, min_orders_per_dispatch=4)
    for pickup_idx in [1, 2, 3]:
        state = state.update(torch.tensor([pickup_idx]))
        state = state.update(torch.tensor([pickup_idx + n_orders]))

    mask = state.get_mask()
    depot_masked_before = bool(mask[0, 0, 0].item())
    print(f'完成3单后 depot masked={depot_masked_before}')
    assert depot_masked_before is True, '未满4单时 depot 必须被硬屏蔽'

    state = state.update(torch.tensor([4]))
    state = state.update(torch.tensor([8]))
    mask = state.get_mask()
    depot_masked_after = bool(mask[0, 0, 0].item())
    print(f'完成4单后 depot masked={depot_masked_after}')
    assert depot_masked_after is False, '达到4单后 depot 应可恢复可行'


def dead_end_reject_under_min_dispatch_regression():
    print('\n' + '=' * 60)
    print('min-orders dead-end reject 可用性回归测试')
    print('=' * 60)

    old_hard = Config.HARD_CARGO_PICKUP_TIMEWINDOW
    Config.HARD_CARGO_PICKUP_TIMEWINDOW = True
    try:
        n_orders = 2
        depot = torch.tensor([[0.2, 0.2]], dtype=torch.float)
        pickup = torch.tensor([[0.24, 0.24], [0.80, 0.80]], dtype=torch.float)
        delivery = torch.tensor([[0.24, 0.34], [0.82, 0.82]], dtype=torch.float)
        loc = torch.cat([pickup, delivery], dim=0).unsqueeze(0)
        node_type = torch.zeros(1, 2 * n_orders, dtype=torch.float)
        demand_passenger = torch.zeros(1, 2 * n_orders, dtype=torch.float)
        demand_cargo = torch.zeros(1, 2 * n_orders, dtype=torch.float)
        demand_cargo[0, :n_orders] = 0.2
        demand_cargo[0, n_orders:] = -0.2
        time_windows = torch.tensor([[[10.0, 16.0], [10.0, 10.02], [10.0, 16.0], [10.0, 16.0]]], dtype=torch.float)
        batch = {
            'depot': depot,
            'loc': loc,
            'node_type': node_type,
            'demand_passenger': demand_passenger,
            'demand_cargo': demand_cargo,
            'time_windows': time_windows,
        }
        state = StateMCVRPPDTW.initialize(batch, min_orders_per_dispatch=4, allow_reject=True)
        state = state.update(torch.tensor([1]))
        state = state.update(torch.tensor([3]))
        mask, debug = state.get_mask(return_debug=True)
        reject_available = bool((~mask[0, 0, state.reject_index]).item())
        depot_masked = bool(mask[0, 0, 0].item())
        print(f'reject_available={reject_available}, depot_masked={depot_masked}, inroute_dead_end={debug["diag_reject_dead_end_inroute_available"].item():.1f}')
        assert depot_masked is True, '未满4单且仍在路上时 depot 应被屏蔽'
        assert reject_available is True, 'dead-end 场景必须放开 reject'
    finally:
        Config.HARD_CARGO_PICKUP_TIMEWINDOW = old_hard


def dry_run():
    basic_dry_run()
    shared_mask_regression()
    viability_fallback_regression()
    pickup_time_update_regression()
    pickup_commitment_next_delivery_equivalence_regression()
    attention_shrink_pomo_regression()
    urgency_bias_changes_greedy_choice_regression()
    urgency_bias_preserves_mask_semantics_regression()
    urgency_bias_training_path_slack_available_regression()
    pomo_baseline_mode_regression()
    business_priority_regression()
    cargo_pickup_hard_timewindow_regression()
    cargo_piecewise_delay_cost_regression()
    min_orders_per_dispatch_regression()
    dead_end_reject_under_min_dispatch_regression()
    print('\nDry-run 验证完成！')


if __name__ == '__main__':
    dry_run()
