import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from problem_mcvrptw_v2 import MCVRPPDTW, MCVRPPDTWDataset, Config
from state_mcvrptw_v2 import StateMCVRPPDTW

def dry_run():
    print("="*60)
    print("VRP 重构逻辑 Dry-run 验证")
    print("="*60)
    
    batch_size = 1
    n_orders = 5
    n_nodes = 2 * n_orders
    
    # 1. 初始化数据集
    dataset = MCVRPPDTWDataset(num_samples=batch_size, graph_size=n_orders)
    sample = dataset[0]
    
    # 模拟输入字典
    input_data = {
        'depot': sample['depot'].unsqueeze(0),
        'loc': sample['loc'].unsqueeze(0),
        'node_type': sample['node_type'].unsqueeze(0),
        'demand_passenger': sample['demand_passenger'].unsqueeze(0),
        'demand_cargo': sample['demand_cargo'].unsqueeze(0),
        'time_windows': sample['time_windows'].unsqueeze(0)
    }
    
    # 2. 初始化状态
    state = StateMCVRPPDTW.initialize(input_data)
    print(f"初始状态: 时间={state.current_time.item():.2f}, 车辆={state.used_vehicles.item()}")
    
    # 3. 验证掩码逻辑
    mask = state.get_mask()
    last_node = 2 * n_orders  # 最后一个 delivery 节点索引（因为不再存在“充电站”额外节点）
    print(f"初始掩码: 节点0(换车)={mask[0, 0, 0].item()}, 节点{last_node}(最后一个delivery)={mask[0, 0, last_node].item()}")
    print("预期: 初始时由于没有任务完成且有任务可做，回库点应被屏蔽。")
    
    # 4. 执行一些步骤
    # 选取一个pickup点
    pickup_idx = 1
    state = state.update(torch.tensor([pickup_idx]))
    print(f"\n步骤1 (访问Pickup {pickup_idx}): 时间={state.current_time.item():.2f}, 乘客容量={state.used_capacity_passenger.item():.2f}")
    
    # 验证禁止带货回库
    mask = state.get_mask()
    print(f"带货掩码: 节点0(换车)={mask[0, 0, 0].item()}, 节点{last_node}(最后一个delivery)={mask[0, 0, last_node].item()}")
    print("预期: 带货时回库点必须被屏蔽 (True).")
    
    # 完成该任务的delivery
    delivery_idx = pickup_idx + n_orders
    state = state.update(torch.tensor([delivery_idx]))
    print(f"\n步骤2 (访问Delivery {delivery_idx}): 时间={state.current_time.item():.2f}, 乘客容量={state.used_capacity_passenger.item():.2f}")
    
    # 验证此时可以回库
    mask = state.get_mask()
    print(f"空载掩码: 节点0(换车)={mask[0, 0, 0].item()}, 节点{last_node}(最后一个delivery)={mask[0, 0, last_node].item()}")
    
    # 5. 验证换车逻辑 (节点0)
    state = state.update(torch.tensor([0]))
    print(f"\n步骤3 (返回车场-换新车): 时间={state.current_time.item():.2f}, 车辆={state.used_vehicles.item()}")
    print("预期: 时间应重置为10.0，车辆数应增加 1（当前实现记已启用车辆数）。")
    
    # 6. （已移除充电站节点）当前环境中只有 depot(0) 用于换车分隔，不再验证充电逻辑。
    
    # 7. 验证成本计算
    # 构造一个简单的路径序列
    # pi: [0, 1, 1+n_orders, 0, 2, 2+n_orders, 0]
    pi = torch.tensor([[0, 1, 1+n_orders, 0, 2, 2+n_orders, 0]])
    costs, details = MCVRPPDTW.get_costs(input_data, pi, return_details=True)
    
    print(f"\n最终成本验证:")
    print(f"  总成本: {costs.item():.2f} 元")
    print(f"  车辆成本: {details['vehicle_cost'].item():.2f} 元")
    print(f"  使用车辆数: {details['used_vehicles'].item()}")
    print(f"  能耗成本: {details['energy_cost'].item():.2f} 元")
    print(f"  时间窗超时惩罚: {details['trip_overtime_penalty'].item():.2f} 元")
    
    print("\nDry-run 验证完成！")

if __name__ == '__main__':
    dry_run()
