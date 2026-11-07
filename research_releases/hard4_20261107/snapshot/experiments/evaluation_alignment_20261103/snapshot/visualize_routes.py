"""
混合客货运输车辆路径可视化
Mixed Passenger-Cargo Vehicle Routing Visualization

生成论文级空间路径示意图，展示：
- 乘客节点 (Passenger Nodes) - 蓝色圆点
- 货物节点 (Cargo Nodes) - 橙色方块
- 仓库 (Depot) - 红色五角星
- 车辆路径 - 不同颜色折线+方向箭头

输出:
- route_25.png  (25订单)
- route_50.png  (50订单)
- route_100.png (100订单)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os

# 核心模块
from problem_mcvrptw_v2 import MCVRPPDTW, MCVRPPDTWDataset, Config
from nets.attention_model import AttentionModel, set_decode_type

# ============================================================
# 全局配置
# ============================================================

# 路径颜色调色板 (高区分度)
ROUTE_COLORS = [
    '#E41A1C',  # 红色
    '#377EB8',  # 蓝色
    '#4DAF4A',  # 绿色
    '#984EA3',  # 紫色
    '#FF7F00',  # 橙色
    '#FFFF33',  # 黄色
    '#A65628',  # 棕色
    '#F781BF',  # 粉色
    '#999999',  # 灰色
    '#66C2A5',  # 青色
    '#FC8D62',  # 珊瑚色
    '#8DA0CB',  # 淡蓝色
    '#E78AC3',  # 淡粉色
    '#A6D854',  # 黄绿色
    '#FFD92F',  # 金色
]

# 节点样式
PASSENGER_COLOR = '#1E90FF'   # 道奇蓝
CARGO_COLOR = '#FF8C00'       # 深橙色
DEPOT_COLOR = '#DC143C'       # 深红色

# 字体配置 (论文级)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


# ============================================================
# 工具函数
# ============================================================

def collate_fn(batch):
    """Collate batch data into tensors"""
    keys = batch[0].keys()
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        if torch.is_tensor(batch[0][key]) else batch[0][key]
        for key in keys
    }


def load_model_for_size(graph_size, device):
    """加载指定规模的训练模型"""
    model_paths = [
        f'outputs/pomo_n{graph_size}_optimized/model_best.pt',
        f'outputs/pomo_n{graph_size}/model_best.pt',
        f'outputs/pomo_n{graph_size}_optimized/model_final.pt',
        f'outputs/pomo_n{graph_size}/model_final.pt'
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print(f"✗ 未找到 {graph_size} 订单的模型文件")
        return None

    print(f"  加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    if 'args' in checkpoint and checkpoint['args'] is not None:
        args_dict = checkpoint['args']
        model_params = {
            'embedding_dim': args_dict.get('embedding_dim', 256),
            'hidden_dim': args_dict.get('hidden_dim', 256),
            'n_encode_layers': args_dict.get('n_encode_layers', 6),
            'tanh_clipping': args_dict.get('tanh_clipping', 10.),
            'normalization': args_dict.get('normalization', 'batch'),
            'n_heads': args_dict.get('n_heads', 8),
            'problem': MCVRPPDTW
        }
    else:
        model_params = {
            'embedding_dim': 256,
            'hidden_dim': 256,
            'n_encode_layers': 6,
            'tanh_clipping': 10.,
            'normalization': 'batch',
            'n_heads': 8,
            'problem': MCVRPPDTW
        }

    model = AttentionModel(**model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def get_routes_from_model(model, batch, device):
    """从模型获取路径"""
    model.eval()
    set_decode_type(model, 'greedy')

    with torch.no_grad():
        batch_input = {k: v.to(device) if torch.is_tensor(v) else v
                       for k, v in batch.items()}
        cost, _, pi = model(batch_input, return_pi=True)
        route = pi[0].cpu().numpy()

    # 拆分路径
    sub_routes = []
    current_sub = []
    for node_idx in route:
        if node_idx == 0:
            if len(current_sub) > 0:
                sub_routes.append(current_sub)
                current_sub = []
        else:
            current_sub.append(node_idx)
    if len(current_sub) > 0:
        sub_routes.append(current_sub)

    return sub_routes


# ============================================================
# 路径绘制
# ============================================================

def plot_route_map(batch, routes, num_orders, save_path):
    """
    绘制混合客货运输车辆路径示意图

    参数:
    - batch: 数据批次
    - routes: 路径列表
    - num_orders: 订单数量
    - save_path: 保存路径
    """
    # 提取数据
    depot = batch['depot'][0].cpu().numpy() * Config.AREA_SIZE  # 转换为km
    locs = batch['loc'][0].cpu().numpy() * Config.AREA_SIZE     # 转换为km
    node_type = batch['node_type'][0].cpu().numpy()
    n_orders = batch['n_orders']

    # 图像尺寸
    fig, ax = plt.subplots(figsize=(12, 10))

    # 设置坐标轴范围 (稍微扩大边界)
    padding = Config.AREA_SIZE * 0.08
    ax.set_xlim(-padding, Config.AREA_SIZE + padding)
    ax.set_ylim(-padding, Config.AREA_SIZE + padding)

    # 绘制网格
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # 绘制边界框
    boundary = plt.Rectangle((0, 0), Config.AREA_SIZE, Config.AREA_SIZE,
                             fill=False, edgecolor='gray',
                             linestyle='--', linewidth=1.5, alpha=0.5)
    ax.add_patch(boundary)

    # 统计节点类型
    passenger_pickup_indices = []
    passenger_delivery_indices = []
    cargo_pickup_indices = []
    cargo_delivery_indices = []

    for i in range(n_orders * 2):
        if i < n_orders:  # Pickup nodes
            if node_type[i] == 1:  # 乘客
                passenger_pickup_indices.append(i)
            else:  # 货物
                cargo_pickup_indices.append(i)
        else:  # Delivery nodes
            if node_type[i] == 1:  # 乘客
                passenger_delivery_indices.append(i)
            else:  # 货物
                cargo_delivery_indices.append(i)

    # 绘制车辆路径
    legend_routes = []
    for route_idx, route in enumerate(routes):
        if len(route) == 0:
            continue

        color = ROUTE_COLORS[route_idx % len(ROUTE_COLORS)]

        # 构建完整路径坐标 (depot -> nodes -> depot)
        path_x = [depot[0]]
        path_y = [depot[1]]

        for node_idx in route:
            real_idx = node_idx - 1
            path_x.append(locs[real_idx, 0])
            path_y.append(locs[real_idx, 1])

        # 返回depot
        path_x.append(depot[0])
        path_y.append(depot[1])

        # 绘制路径线
        ax.plot(path_x, path_y, '-', color=color, linewidth=1.8,
                alpha=0.8, zorder=2)

        # 绘制方向箭头
        for i in range(len(path_x) - 1):
            # 计算箭头位置 (中点偏后)
            mid_x = path_x[i] * 0.4 + path_x[i+1] * 0.6
            mid_y = path_y[i] * 0.4 + path_y[i+1] * 0.6

            # 计算方向
            dx = path_x[i+1] - path_x[i]
            dy = path_y[i+1] - path_y[i]

            # 归一化
            length = np.sqrt(dx**2 + dy**2)
            if length > 0.3:  # 只在足够长的线段上画箭头
                dx_norm = dx / length
                dy_norm = dy / length

                # 箭头大小
                arrow_size = 0.25

                ax.annotate('',
                            xy=(mid_x + dx_norm * arrow_size * 0.5,
                                mid_y + dy_norm * arrow_size * 0.5),
                            xytext=(mid_x - dx_norm * arrow_size * 0.5,
                                    mid_y - dy_norm * arrow_size * 0.5),
                            arrowprops=dict(arrowstyle='->',
                                            color=color,
                                            lw=1.5,
                                            mutation_scale=12),
                            zorder=3)

        legend_routes.append(Line2D([0], [0], color=color, linewidth=2,
                                    label=f'Route {route_idx + 1}'))

    # 绘制乘客节点 (蓝色圆点)
    passenger_indices = passenger_pickup_indices + passenger_delivery_indices
    if passenger_indices:
        passenger_x = [locs[i, 0] for i in passenger_indices]
        passenger_y = [locs[i, 1] for i in passenger_indices]
        ax.scatter(passenger_x, passenger_y,
                   c=PASSENGER_COLOR, s=80, marker='o',
                   edgecolors='black', linewidths=0.8,
                   label='Passenger Nodes', zorder=5)

    # 绘制货物节点 (橙色方块)
    cargo_indices = cargo_pickup_indices + cargo_delivery_indices
    if cargo_indices:
        cargo_x = [locs[i, 0] for i in cargo_indices]
        cargo_y = [locs[i, 1] for i in cargo_indices]
        ax.scatter(cargo_x, cargo_y,
                   c=CARGO_COLOR, s=100, marker='s',
                   edgecolors='black', linewidths=0.8,
                   label='Cargo Nodes', zorder=5)

    # 绘制Depot (红色五角星)
    ax.scatter(depot[0], depot[1],
               c=DEPOT_COLOR, s=300, marker='*',
               edgecolors='black', linewidths=1,
               label='Depot', zorder=6)

    # 坐标轴标签
    ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance (km)', fontsize=12, fontweight='bold')

    # 标题
    ax.set_title(f'Mixed Passenger-Cargo Vehicle Routing\n'
                 f'{num_orders} Orders ({num_orders*2} Nodes) - {len(routes)} Vehicles',
                 fontsize=14, fontweight='bold', pad=15)

    # 等比例
    ax.set_aspect('equal')

    # 创建图例
    # 节点图例
    node_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PASSENGER_COLOR,
               markersize=10, markeredgecolor='black', markeredgewidth=0.8,
               label='Passenger Nodes'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CARGO_COLOR,
               markersize=10, markeredgecolor='black', markeredgewidth=0.8,
               label='Cargo Nodes'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=DEPOT_COLOR,
               markersize=15, markeredgecolor='black', markeredgewidth=1,
               label='Depot'),
    ]

    # 合并图例
    all_legend = legend_routes + node_legend

    # 放置图例 (右侧)
    ax.legend(handles=all_legend,
              loc='upper left',
              bbox_to_anchor=(1.02, 1),
              frameon=True,
              fancybox=True,
              shadow=True,
              fontsize=9)

    # 统计信息文本
    n_passenger = len(passenger_indices)
    n_cargo = len(cargo_indices)
    info_text = f'Passenger: {n_passenger} | Cargo: {n_cargo}'
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 布局优化
    plt.tight_layout()

    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"✓ 路径图已保存: {save_path}")


# ============================================================
# 主程序
# ============================================================

def run_visualization():
    """主入口: 生成所有规模的路径图"""

    print("=" * 70)
    print("Mixed Passenger-Cargo Vehicle Routing Visualization")
    print("混合客货运输车辆路径可视化生成器")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 创建输出目录
    output_dir = 'outputs/route_maps'
    os.makedirs(output_dir, exist_ok=True)

    # 问题规模
    problem_sizes = [25, 50, 100]

    for graph_size in problem_sizes:
        print(f"\n{'='*60}")
        print(f"处理 {graph_size} 订单问题 ({graph_size*2} 节点)")
        print("=" * 60)

        # 加载模型
        model = load_model_for_size(graph_size, device)
        if model is None:
            continue

        # 生成测试数据 (固定种子)
        dataset = MCVRPPDTWDataset(
            num_samples=1,
            graph_size=graph_size,
            seed=2024
        )
        batch = collate_fn([dataset[0]])

        # 获取路径
        print("  生成路径...")
        routes = get_routes_from_model(model, batch, device)

        print(f"  车辆数量: {len(routes)}")
        print(f"  路径节点: {[len(r) for r in routes]}")

        # 绘制路径图
        save_path = os.path.join(output_dir, f'route_{graph_size}.png')
        plot_route_map(batch, routes, graph_size, save_path)

    print(f"\n{'='*70}")
    print("所有路径图生成完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 70)


def generate_single_route_map(graph_size, seed=2024, save_path=None):
    """
    生成单个规模的路径图 (API接口)

    参数:
    - graph_size: 订单数量 (25/50/100)
    - seed: 随机种子
    - save_path: 保存路径 (可选)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model_for_size(graph_size, device)
    if model is None:
        return None

    dataset = MCVRPPDTWDataset(num_samples=1, graph_size=graph_size, seed=seed)
    batch = collate_fn([dataset[0]])

    routes = get_routes_from_model(model, batch, device)

    if save_path is None:
        save_path = f'route_{graph_size}.png'

    plot_route_map(batch, routes, graph_size, save_path)

    return routes


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    run_visualization()
