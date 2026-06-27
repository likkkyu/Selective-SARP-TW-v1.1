import os
import sys

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


def _build_test_attention_model(shrink_size):
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
            repeated_batch,
            return_pi=True,
            state_kwargs=state_kwargs,
            return_debug=True,
        )
        cost_shrink, ll_shrink, pi_shrink, debug_shrink = model_shrink(
            repeated_batch,
            return_pi=True,
            state_kwargs=state_kwargs,
            return_debug=True,
        )

    print(f'no_shrink pi shape={tuple(pi_no_shrink.shape)}, shrink pi shape={tuple(pi_shrink.shape)}')
    assert pi_no_shrink.shape == pi_shrink.shape, 'shrink/no-shrink 的 pi shape 不一致'
    assert torch.equal(pi_no_shrink, pi_shrink), 'shrink/no-shrink greedy decode 序列不一致'
    assert torch.allclose(cost_no_shrink, cost_shrink), 'shrink/no-shrink cost 不一致'
    assert torch.allclose(ll_no_shrink, ll_shrink), 'shrink/no-shrink log likelihood 不一致'

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
        assert key in debug_no_shrink and key in debug_shrink, f'缺少 shrink debug key: {key}'
        assert debug_no_shrink[key].shape == debug_shrink[key].shape, f'shrink/no-shrink debug[{key}] shape 不一致'
        assert torch.isfinite(debug_no_shrink[key]).all(), f'no-shrink debug[{key}] 出现非有限值'
        assert torch.isfinite(debug_shrink[key]).all(), f'shrink debug[{key}] 出现非有限值'


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


def dry_run():
    basic_dry_run()
    shared_mask_regression()
    viability_fallback_regression()
    pickup_commitment_next_delivery_equivalence_regression()
    attention_shrink_pomo_regression()
    pomo_baseline_mode_regression()
    business_priority_regression()
    print('\nDry-run 验证完成！')


if __name__ == '__main__':
    dry_run()
