import os
import sys

import torch

# Add current directory to path
sys.path.append(os.getcwd())

from problem_mcvrptw_v2 import MCVRPPDTW, MCVRPPDTWDataset, Config
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


def dry_run():
    basic_dry_run()
    shared_mask_regression()
    viability_fallback_regression()
    print('\nDry-run 验证完成！')


if __name__ == '__main__':
    dry_run()
