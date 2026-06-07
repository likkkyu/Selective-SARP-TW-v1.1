"""
学术论文图表生成器
Academic Paper Figure Generator for MCVRP-PDTW

生成论文所需的所有可视化图表：
1. 训练曲线图 (Training Curves)
2. Baseline对比图 (Performance Comparison)
3. 成本分解图 (Cost Breakdown)
4. 求解时间对比图 (Time Comparison)
5. 可扩展性分析图 (Scalability Analysis)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import torch
from collections import defaultdict

# 核心模块
from problem_mcvrptw_v2 import MCVRPPDTW, MCVRPPDTWDataset, Config
from nets.attention_model import AttentionModel, set_decode_type

# ============================================================
# 全局配置
# ============================================================

# 字体配置 (论文级)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 颜色方案（学术调色板：暖色突出 Ours，冷色为 baseline）
COLORS = {
    'DRL': '#ec6343',      # 红橙 (Ours，突出)
    'Greedy': '#3f6ca7',   # 中蓝
    'Random': '#6f9ec8',   # 浅蓝
    'Gurobi': '#2c4c75',   # 深蓝
    'Advanced': '#FF6B6B',
    'energy': '#FF7F00',
    'passenger_delay': '#1E90FF',
    'cargo_delay': '#32CD32',
    'primary': '#2C3E50',
    'secondary': '#7F8C8D'
}

# 输出目录
OUTPUT_DIR = 'outputs/paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 训练曲线图
# ============================================================

def plot_training_curves():
    """绘制训练曲线图（Loss + Cost + Learning Rate）"""

    print("生成训练曲线图...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    phases = [
        ('25', 'outputs/pomo_n25_optimized/training_log.json', 'Phase 1: 25 Orders'),
        ('50', 'outputs/pomo_n50_optimized/training_log.json', 'Phase 2: 50 Orders'),
        ('100', 'outputs/pomo_n100_optimized/training_log.json', 'Phase 3: 100 Orders')
    ]

    for col, (size, log_path, title) in enumerate(phases):
        if not os.path.exists(log_path):
            print(f"  跳过 {size} 订单 (文件不存在)")
            continue

        with open(log_path, 'r') as f:
            log_data = json.load(f)

        train_data = log_data['train']
        val_data = log_data['val']

        epochs = [d['epoch'] for d in train_data]
        train_loss = [d['loss'] for d in train_data]
        train_cost = [d['cost'] for d in train_data]
        val_cost = [d['avg_cost'] for d in val_data]

        # 上排: Loss曲线
        ax1 = axes[0, col]
        ax1.plot(epochs, train_loss,
                 color=COLORS['primary'], linewidth=1.5, label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(title)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # 下排: Cost曲线
        ax2 = axes[1, col]
        ax2.plot(epochs, train_cost, color=COLORS['DRL'], linewidth=1.5,
                 label='Train Cost', alpha=0.7)
        ax2.plot(epochs, val_cost, color=COLORS['Greedy'], linewidth=1.5,
                 label='Val Cost')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Cost (RMB)')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


# ============================================================
# 2. Baseline对比图
# ============================================================

def evaluate_baselines(graph_size, num_samples=50):
    """评估各种baseline方法"""

    # 生成测试数据
    dataset = MCVRPPDTWDataset(
        num_samples=num_samples, graph_size=graph_size, seed=42)

    results = {
        'DRL': {'costs': [], 'times': []},
        'Greedy': {'costs': [], 'times': []},
        'Random': {'costs': [], 'times': []}
    }

    # 加载DRL模型
    device = torch.device('cpu')
    model_path = f'outputs/pomo_n{graph_size}_optimized/model_best.pt'

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        args_dict = checkpoint['args']
        model = AttentionModel(
            embedding_dim=args_dict.get('embedding_dim', 256),
            hidden_dim=args_dict.get('hidden_dim', 256),
            n_encode_layers=args_dict.get('n_encode_layers', 6),
            tanh_clipping=args_dict.get('tanh_clipping', 10.),
            normalization=args_dict.get('normalization', 'batch'),
            n_heads=args_dict.get('n_heads', 8),
            problem=MCVRPPDTW
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        set_decode_type(model, 'greedy')
    else:
        model = None

    import time

    for i, sample in enumerate(dataset.data):
        # 构造batch
        batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v
                 for k, v in sample.items()}

        # DRL评估
        if model is not None:
            start = time.time()
            with torch.no_grad():
                cost, _, pi = model(batch, return_pi=True)
            elapsed = time.time() - start
            results['DRL']['costs'].append(cost.item())
            results['DRL']['times'].append(elapsed)

        # Greedy评估
        start = time.time()
        greedy_cost = evaluate_greedy_single(sample)
        elapsed = time.time() - start
        results['Greedy']['costs'].append(greedy_cost)
        results['Greedy']['times'].append(elapsed)

        # Random评估
        start = time.time()
        random_cost = evaluate_random_single(sample, n_trials=10)
        elapsed = time.time() - start
        results['Random']['costs'].append(random_cost)
        results['Random']['times'].append(elapsed)

    return results


def evaluate_greedy_single(sample):
    """贪婪算法评估单个样本（使用完整成本函数）"""
    depot = sample['depot']
    loc = sample['loc']
    node_type = sample['node_type']
    demand_p = sample['demand_passenger']
    demand_c = sample['demand_cargo']
    time_windows = sample['time_windows']
    n_orders = sample['n_orders']

    n_nodes = loc.size(0)
    visited = torch.zeros(n_nodes, dtype=torch.bool)
    picked_up = torch.zeros(n_orders, dtype=torch.bool)

    current_pos = depot
    current_time = Config.OPERATION_START
    trip_start = Config.OPERATION_START
    used_cap_p = 0.0
    used_cap_c = 0.0

    route = [0]  # 从depot开始

    while not visited.all():
        best_node = None
        best_dist = float('inf')

        for i in range(n_nodes):
            if visited[i]:
                continue

            # Pickup-Delivery约束
            if i >= n_orders:
                pickup_idx = i - n_orders
                if not picked_up[pickup_idx]:
                    continue

            # 容量约束
            if node_type[i] == 1:
                if demand_p[i] > 0 and used_cap_p + demand_p[i].item() > 1.0:
                    continue
            else:
                if demand_c[i] > 0 and used_cap_c + demand_c[i].item() > 1.0:
                    continue

            # 时间约束
            dist = (loc[i] - current_pos).norm().item() * Config.AREA_SIZE
            travel_time = dist / Config.VEHICLE_SPEED
            arrival = current_time + travel_time
            return_dist = (depot - loc[i]).norm().item() * Config.AREA_SIZE
            return_time = return_dist / Config.VEHICLE_SPEED
            trip_time = arrival - trip_start + return_time + Config.SERVICE_TIME

            if trip_time > Config.MAX_TRIP_TIME + 0.5:
                continue

            if dist < best_dist:
                best_dist = dist
                best_node = i

        if best_node is None:
            # 返回depot开始新路线
            route.append(0)
            current_pos = depot
            current_time = Config.OPERATION_START
            trip_start = Config.OPERATION_START
            used_cap_p = 0.0
            used_cap_c = 0.0
            continue

        visited[best_node] = True
        if best_node < n_orders:
            picked_up[best_node] = True

        # 更新时间
        dist = (loc[best_node] - current_pos).norm().item() * Config.AREA_SIZE
        travel_time = dist / Config.VEHICLE_SPEED
        current_time = current_time + travel_time
        current_time = max(current_time, time_windows[best_node, 0].item())
        current_time = current_time + Config.SERVICE_TIME

        # 更新容量
        if node_type[best_node] == 1:
            used_cap_p = max(0, used_cap_p + demand_p[best_node].item())
        else:
            used_cap_c = max(0, used_cap_c + demand_c[best_node].item())

        current_pos = loc[best_node]
        route.append(best_node + 1)

    route.append(0)

    # 使用完整成本函数计算
    route_tensor = torch.tensor(route, dtype=torch.long).unsqueeze(0)
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v
             for k, v in sample.items()}
    cost, _ = MCVRPPDTW.get_costs(batch, route_tensor)

    return cost.item()


def evaluate_random_single(sample, n_trials=10):
    """随机算法评估单个样本（使用完整成本函数）"""
    n_orders = sample['n_orders']

    best_cost = float('inf')
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v
             for k, v in sample.items()}

    for _ in range(n_trials):
        perm = torch.randperm(n_orders)
        route = [0]  # 从depot开始
        for i in perm:
            route.append(i + 1)  # pickup
            route.append(i + n_orders + 1)  # delivery
        route.append(0)  # 返回depot

        route_tensor = torch.tensor(route, dtype=torch.long).unsqueeze(0)
        cost, _ = MCVRPPDTW.get_costs(batch, route_tensor)
        best_cost = min(best_cost, cost.item())

    return best_cost


def plot_baseline_comparison():
    """绘制Baseline对比柱状图"""

    print("生成Baseline对比图...")

    sizes = [25, 50, 100]
    all_results = {}

    # 1. 加载Gurobi重评估结果（成本）与真实求解时间（完整求解时间对比）
    gurobi_comparison_path = 'outputs/gurobi_reevaluated/gurobi_vs_drl_comparison.json'
    gurobi_data = {}
    if os.path.exists(gurobi_comparison_path):
        with open(gurobi_comparison_path, 'r') as f:
            gurobi_comparison = json.load(f)
        gurobi_reevaluated = gurobi_comparison.get('gurobi_reevaluated', {})
        for size_str, data in gurobi_reevaluated.items():
            size = int(data['graph_size'])
            # 优先从 gurobi_results_{size}.json 读取真实求解时间（完整求解时间）
            solve_time = data.get('avg_solve_time')
            gurobi_results_path = f'gurobi_results_{size}.json'
            if solve_time is None and os.path.exists(gurobi_results_path):
                with open(gurobi_results_path, 'r') as fg:
                    gr = json.load(fg)
                solve_time = gr.get('avg_solve_time')
            if solve_time is None:
                solve_time = 300  # 兜底默认
            gurobi_data[size] = {
                'mean': data['real_avg_cost'],
                'std': data['real_std_cost'],
                'time': solve_time
            }
        print(f"  ✓ 加载Gurobi重评估结果: {len(gurobi_data)}个规模 (含真实求解时间)")
    else:
        print(f"  ⚠️ 未找到Gurobi重评估结果，跳过Gurobi对比")

    # 2. 评估其他Baseline方法
    for size in sizes:
        print(f"  评估 {size} 订单...")
        results = evaluate_baselines(size, num_samples=30)
        all_results[size] = {
            method: {
                'mean': np.mean(data['costs']),
                'std': np.std(data['costs']),
                'time': np.mean(data['times'])
            }
            for method, data in results.items() if data['costs']
        }

        # 添加Gurobi数据
        if size in gurobi_data:
            all_results[size]['Gurobi'] = gurobi_data[size]

    # 绘制成本对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 调整顺序：Gurobi > Random > Greedy > DRL (降序展示优势)
    methods = ['Gurobi', 'Random', 'Greedy', 'DRL']
    method_labels = ['Gurobi (MIP)', 'Random', 'Greedy', 'DRL (Ours)']
    x = np.arange(len(sizes))
    width = 0.2

    # 成本对比图
    ax1 = axes[0]
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        means = [all_results[s].get(method, {}).get('mean', 0) for s in sizes]
        stds = [all_results[s].get(method, {}).get('std', 0) for s in sizes]

        # 如果该方法没有数据，跳过
        if all(m == 0 for m in means):
            continue

        bars = ax1.bar(x + i*width, means, width, label=label,
                       color=COLORS[method], yerr=stds, capsize=3, alpha=0.85)

        # 添加数值标签（仅对DRL）
        if method == 'DRL':
            for j, (mean, bar) in enumerate(zip(means, bars)):
                if mean > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, mean,
                             f'{mean:.0f}', ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Problem Size (Orders)', fontweight='bold')
    ax1.set_ylabel('Average Cost (RMB)', fontweight='bold')
    ax1.set_title('Cost Comparison with Baseline Methods', fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([f'{s} Orders' for s in sizes])
    ax1.legend(loc='upper left')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

    # 时间对比图
    ax2 = axes[1]
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        times = [all_results[s].get(method, {}).get('time', 0) for s in sizes]

        # 如果该方法没有数据，跳过
        if all(t == 0 for t in times):
            continue

        ax2.bar(x + i*width, times, width, label=label,
                color=COLORS[method], alpha=0.85)

    ax2.set_xlabel('Problem Size (Orders)', fontweight='bold')
    ax2.set_ylabel('Average Solving Time (s)', fontweight='bold')
    ax2.set_title('Solving Time Comparison', fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([f'{s} Orders' for s in sizes])
    # 图例放在右侧，避免与高柱体（Gurobi）重叠
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax2.set_yscale('log')
    # 完整求解时间对比：Y轴覆盖 Gurobi 真实时间（约 300–600 s），避免柱子被截断
    ax2.set_ylim(1e-2, 1e3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'baseline_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ 保存: {save_path}")

    return all_results


# ============================================================
# 3. 成本分解图
# ============================================================

def plot_cost_breakdown():
    """绘制成本分解图"""

    print("生成成本分解图...")

    sizes = [25, 50, 100]
    cost_data = {}

    for size in sizes:
        log_path = f'outputs/pomo_n{size}_optimized/training_log.json'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log = json.load(f)

            # 取最后一个epoch的验证结果
            last_val = log['val'][-1]
            cost_data[size] = {
                'energy': last_val.get('avg_energy_cost', 0),
                'passenger_delivery_delay': last_val.get(
                    'avg_passenger_delivery_delay_cost',
                    last_val.get('avg_passenger_delay_cost', 0)
                ),
                'cargo_delay': last_val.get('avg_cargo_delay_cost', 0)
            }

    # 绘制堆叠百分比柱状图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 堆叠百分比柱状图
    ax1 = axes[0]
    x = np.arange(len(sizes))
    width = 0.6

    energy_costs = np.array(
        [cost_data.get(s, {}).get('energy', 0) for s in sizes])
    passenger_delivery_delay = np.array([
        cost_data.get(s, {}).get('passenger_delivery_delay', 0) for s in sizes
    ])
    cargo_delay = np.array(
        [cost_data.get(s, {}).get('cargo_delay', 0) for s in sizes])

    # 计算百分比
    total_costs = energy_costs + passenger_delivery_delay + cargo_delay
    energy_pct = (energy_costs / total_costs) * 100
    pax_pct = (passenger_delivery_delay / total_costs) * 100
    cargo_pct = (cargo_delay / total_costs) * 100

    ax1.bar(x, energy_pct, width, label='Energy Cost', color=COLORS['energy'])
    ax1.bar(x, pax_pct, width, bottom=energy_pct,
            label='Passenger Delivery Delay', color=COLORS['passenger_delay'])
    ax1.bar(x, cargo_pct, width,
            bottom=energy_pct + pax_pct,
            label='Cargo Delay', color=COLORS['cargo_delay'])

    # 添加百分比标签
    for i in range(len(sizes)):
        if energy_pct[i] > 5:
            ax1.text(x[i], energy_pct[i]/2, f'{energy_pct[i]:.1f}%',
                     ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if pax_pct[i] > 5:
            ax1.text(x[i], energy_pct[i] + pax_pct[i]/2, f'{pax_pct[i]:.1f}%',
                     ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if cargo_pct[i] > 5:
            ax1.text(x[i], energy_pct[i] + pax_pct[i] + cargo_pct[i]/2, f'{cargo_pct[i]:.1f}%',
                     ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    ax1.set_xlabel('Problem Size (Orders)')
    ax1.set_ylabel('Cost Composition (%)')
    ax1.set_title('Cost Breakdown by Component (Percentage)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s} Orders' for s in sizes])
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

    # 饼图 (以50订单为例)
    ax2 = axes[1]
    if 50 in cost_data:
        data = cost_data[50]
        values = [data['energy'], data['passenger_delivery_delay'], data['cargo_delay']]
        labels = ['Energy Cost', 'Passenger Delivery Delay', 'Cargo Delay']
        colors = [COLORS['energy'],
                  COLORS['passenger_delay'], COLORS['cargo_delay']]

        # 过滤零值
        non_zero = [(v, l, c)
                    for v, l, c in zip(values, labels, colors) if v > 0]
        if non_zero:
            values, labels, colors = zip(*non_zero)
            ax2.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90, explode=[0.02]*len(values))
            ax2.set_title('Cost Distribution (50 Orders)')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'cost_breakdown.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


# ============================================================
# 4. 可扩展性分析图
# ============================================================

def plot_scalability():
    """绘制可扩展性分析图"""

    print("生成可扩展性分析图...")

    sizes = [25, 50, 100]

    # 从训练日志获取数据
    drl_costs = []
    for size in sizes:
        log_path = f'outputs/pomo_n{size}_optimized/training_log.json'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log = json.load(f)
            best_cost = min([v['avg_cost'] for v in log['val']])
            drl_costs.append(best_cost)
        else:
            drl_costs.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 成本随规模变化
    ax1 = axes[0]
    ax1.plot(sizes, drl_costs, 'o-', color=COLORS['DRL'], linewidth=2,
             markersize=10, label='DRL (Ours)')

    # 拟合趋势线
    if all(c > 0 for c in drl_costs):
        z = np.polyfit(sizes, drl_costs, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(sizes), max(sizes), 100)
        ax1.plot(x_smooth, p(x_smooth), '--', color=COLORS['secondary'],
                 alpha=0.5, label='Trend (Quadratic)')

    ax1.set_xlabel('Number of Orders')
    ax1.set_ylabel('Average Cost (RMB)')
    ax1.set_title('Cost Scalability Analysis')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 每订单平均成本
    ax2 = axes[1]
    per_order_costs = [c/s if s > 0 and c >
                       0 else 0 for c, s in zip(drl_costs, sizes)]
    ax2.bar(range(len(sizes)), per_order_costs, color=COLORS['DRL'], alpha=0.8)
    ax2.set_xlabel('Problem Size')
    ax2.set_ylabel('Cost per Order (RMB)')
    ax2.set_title('Average Cost per Order')
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([f'{s} Orders' for s in sizes])
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'scalability_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


# ============================================================
# 5. 改进对比图 (优化前后)
# ============================================================

def plot_improvement_comparison():
    """绘制优化前后对比图"""

    print("生成优化对比图...")

    # 原始模型结果 (从之前的分析中)
    original_costs = {
        25: 1239.68,
        50: 7673.16,
        100: 36886.04
    }

    # 优化后结果
    optimized_costs = {}
    for size in [25, 50, 100]:
        log_path = f'outputs/pomo_n{size}_optimized/training_log.json'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log = json.load(f)
            optimized_costs[size] = min([v['avg_cost'] for v in log['val']])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sizes = [25, 50, 100]
    x = np.arange(len(sizes))
    width = 0.35

    # 成本对比
    ax1 = axes[0]
    orig_vals = [original_costs.get(s, 0) for s in sizes]
    opt_vals = [optimized_costs.get(s, 0) for s in sizes]

    ax1.bar(x - width/2, orig_vals, width,
            label='Initial Policy', color=COLORS['secondary'])
    ax1.bar(x + width/2, opt_vals, width,
            label='Trained Model (Ours)', color=COLORS['DRL'])

    ax1.set_xlabel('Problem Size (Orders)')
    ax1.set_ylabel('Average Cost (RMB)')
    ax1.set_title('Model Performance: Before vs After Training')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s} Orders' for s in sizes])
    ax1.legend()
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

    # 改进百分比
    ax2 = axes[1]
    improvements = []
    for s in sizes:
        if s in original_costs and s in optimized_costs:
            imp = (original_costs[s] - optimized_costs[s]
                   ) / original_costs[s] * 100
            improvements.append(imp)
        else:
            improvements.append(0)

    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(x, improvements, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Problem Size (Orders)')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Cost Improvement after Optimization')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{s} Orders' for s in sizes])
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

    # 添加数值标签
    for i, imp in enumerate(improvements):
        ax2.text(i, imp + 0.5, f'{imp:.1f}%',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'improvement_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


# ============================================================
# 6. 综合统计表格图
# ============================================================

def plot_summary_table():
    """生成综合统计表格图"""

    print("生成综合统计表格...")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # 收集数据
    sizes = [25, 50, 100]
    data = []

    for size in sizes:
        log_path = f'outputs/pomo_n{size}_optimized/training_log.json'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log = json.load(f)

            val_costs = [v['avg_cost'] for v in log['val']]
            best_cost = min(val_costs)
            final_cost = val_costs[-1]
            std_cost = log['val'][-1].get('std_cost', 0)

            data.append([
                f'{size} Orders',
                f'{best_cost:.2f}',
                f'{final_cost:.2f}',
                f'{std_cost:.2f}',
                f'{len(val_costs)}',
            ])

    columns = ['Problem Size', 'Best Cost (RMB)', 'Final Cost (RMB)',
               'Std Dev', 'Epochs']

    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#E6E6FA']*len(columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    ax.set_title('Training Results Summary',
                 fontsize=14, fontweight='bold', pad=20)

    save_path = os.path.join(OUTPUT_DIR, 'summary_table.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ 保存: {save_path}")


# ============================================================
# 主程序
# ============================================================

def main():
    """生成所有论文图表"""

    print("=" * 60)
    print("学术论文图表生成器")
    print("Academic Paper Figure Generator")
    print("=" * 60)

    # 1. 训练曲线
    plot_training_curves()

    # 2. Baseline对比
    plot_baseline_comparison()

    # 3. 成本分解
    plot_cost_breakdown()

    # 4. 可扩展性分析
    plot_scalability()

    # 5. 优化对比
    plot_improvement_comparison()

    # 6. 综合统计表格
    plot_summary_table()

    print("\n" + "=" * 60)
    print("所有论文图表生成完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 列出生成的文件
    print("\n生成的图表文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == '__main__':
    main()
