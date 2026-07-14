"""
MCVRP-PDTW 车辆排班甘特图可视化
Gantt Chart Visualization for Multi-Compartment VRP with Pickup-Delivery and Time Windows

功能:
1. 加载训练好的 AttentionModel (DRL)
2. 对单个实例进行 greedy 解码
3. 模拟每辆车的真实执行时间线
4. 生成车辆级执行日志 (execution logs)
5. 绘制论文级甘特图 (Gantt Chart)

时间语义:
- passenger pickup 为硬时间窗
- passenger delivery / cargo delivery 的迟到用红框标注
- 早到仅等待，不计 earliness 成本

输出:
- gantt_25.png  (25个订单)
- gantt_50.png  (50个订单)
- gantt_100.png (100个订单)
"""

import argparse
import csv
import json
import torch
import numpy as np
from datetime import datetime, timedelta
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
except Exception:  # pragma: no cover
    plt = None
    mpatches = None
    Rectangle = None

# 核心模块
from problem_mcvrptw_v2 import MCVRPPDTW, MCVRPPDTWDataset, Config
from nets.attention_model import AttentionModel, set_decode_type

# ============================================================
# 全局配置
# ============================================================

# 基准日期 (用于时间格式化)
BASE_DATE = datetime(2023, 1, 1)

# 甘特图颜色编码 (严格按要求)
COLORS = {
    'move': '#D3D3D3',       # 浅灰色 - 行驶
    'wait': '#FFD700',       # 黄色 - 等待
    'serve_pickup': '#4169E1',    # 蓝色 - 取货服务
    'serve_delivery': '#32CD32',  # 绿色 - 送货服务
    'return': '#696969',     # 深灰色 - 回库
    'tw_background': '#E6E6FA',   # 淡紫色 - 时间窗背景
    'late_border': '#FF0000'      # 红色 - 迟到边框
}

# 字体配置 (论文级)
if plt is not None:
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 9


# ============================================================
# 工具函数
# ============================================================

def hours_to_hhmm(hours_float):
    """将小时浮点数转换为 HH:MM 格式字符串"""
    total_minutes = int(hours_float * 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def format_datetime(hours_float):
    """将小时浮点数转换为 datetime 对象"""
    total_seconds = hours_float * 3600
    midnight = BASE_DATE.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(seconds=total_seconds)


def collate_fn(batch):
    """Collate batch data into tensors"""
    keys = batch[0].keys()
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        if torch.is_tensor(batch[0][key]) else batch[0][key]
        for key in keys
    }


# ============================================================
# 执行日志生成器
# ============================================================

def generate_execution_logs(model, batch, device):
    """
    运行模型并生成执行日志

    返回格式:
    {
        'vehicle_id': int,
        'action': 'move' | 'wait' | 'serve' | 'return',
        'node_id': int,
        'node_type': 'P' | 'D' | 'DEPOT',
        'start_time': float,
        'end_time': float,
        'tw': (float, float),
        'load_change': int
    }
    """
    model.eval()
    set_decode_type(model, 'greedy')

    with torch.no_grad():
        batch_input = {k: v.to(device) if torch.is_tensor(v) else v
                       for k, v in batch.items()}
        cost, _, pi = model(batch_input, return_pi=True)
        route = pi[0].cpu().numpy()

    # 提取数据
    depot = batch['depot'][0].cpu().numpy()
    locs = batch['loc'][0].cpu().numpy()
    node_type = batch['node_type'][0].cpu().numpy()
    dem_p = batch['demand_passenger'][0].cpu().numpy()
    dem_c = batch['demand_cargo'][0].cpu().numpy()
    tw = batch['time_windows'][0].cpu().numpy()
    n_orders = int(batch['n_orders'])

    # 拆分路径为子路径 (以 0 为分隔符)
    # 注意：若模型启用了 reject 动作，pi 中可能出现 > 2*n_orders 的动作索引，这里直接忽略
    sub_routes = []
    current_sub = []
    max_valid_node = 2 * n_orders
    for node_idx in route:
        node_idx = int(node_idx)
        if node_idx == 0:
            if len(current_sub) > 0:
                sub_routes.append(current_sub)
                current_sub = []
            continue

        if 1 <= node_idx <= max_valid_node:
            current_sub.append(node_idx)
    if len(current_sub) > 0:
        sub_routes.append(current_sub)

    # 生成执行日志
    all_logs = []
    speed = Config.VEHICLE_SPEED
    service_time_hours = Config.SERVICE_TIME

    for vehicle_id, sub_route in enumerate(sub_routes):
        if len(sub_route) == 0:
            continue

        current_pos = depot

        # 计算最佳出发时间
        first_node_idx = sub_route[0]
        real_idx = first_node_idx - 1
        first_node_loc = locs[real_idx]
        dist_to_first = np.linalg.norm(
            first_node_loc - depot) * Config.AREA_SIZE
        travel_time_to_first = dist_to_first / speed
        first_tw_start = tw[real_idx][0]
        trip_start_time = max(Config.OPERATION_START,
                              first_tw_start - travel_time_to_first)

        current_time = trip_start_time

        for node_idx in sub_route:
            real_idx = node_idx - 1
            target_pos = locs[real_idx]

            # 计算行驶
            dist_km = np.linalg.norm(
                target_pos - current_pos) * Config.AREA_SIZE
            travel_time = dist_km / speed

            # 添加 move 日志
            move_start = current_time
            move_end = current_time + travel_time

            all_logs.append({
                'vehicle_id': vehicle_id,
                'action': 'move',
                'node_id': node_idx,
                'node_type': 'P' if real_idx < n_orders else 'D',
                'start_time': move_start,
                'end_time': move_end,
                'tw': (tw[real_idx][0], tw[real_idx][1]),
                'load_change': 0
            })

            current_time = move_end

            # 时间窗约束
            tw_start = tw[real_idx][0]
            tw_end = tw[real_idx][1]

            # 如果早到，添加 wait 日志
            if current_time < tw_start:
                wait_start = current_time
                wait_end = tw_start
                all_logs.append({
                    'vehicle_id': vehicle_id,
                    'action': 'wait',
                    'node_id': node_idx,
                    'node_type': 'P' if real_idx < n_orders else 'D',
                    'start_time': wait_start,
                    'end_time': wait_end,
                    'tw': (tw_start, tw_end),
                    'load_change': 0
                })
                current_time = tw_start

            # 添加 serve 日志
            serve_start = current_time
            serve_end = current_time + service_time_hours

            # 计算载重变化
            if real_idx < n_orders:  # Pickup
                if node_type[real_idx] == 1:  # 乘客
                    load_change = int(
                        round(dem_p[real_idx] * Config.PASSENGER_CAPACITY))
                else:  # 货物
                    load_change = int(
                        round(dem_c[real_idx] * Config.CARGO_CAPACITY))
            else:  # Delivery
                if node_type[real_idx] == 1:
                    load_change = int(
                        round(dem_p[real_idx] * Config.PASSENGER_CAPACITY))
                else:
                    load_change = int(
                        round(dem_c[real_idx] * Config.CARGO_CAPACITY))

            all_logs.append({
                'vehicle_id': vehicle_id,
                'action': 'serve',
                'node_id': node_idx,
                'node_type': 'P' if real_idx < n_orders else 'D',
                'start_time': serve_start,
                'end_time': serve_end,
                'tw': (tw_start, tw_end),
                'load_change': load_change
            })

            current_time = serve_end
            current_pos = target_pos

        # 添加 return 日志 (回库)
        dist_home = np.linalg.norm(depot - current_pos) * Config.AREA_SIZE
        return_time = dist_home / speed

        all_logs.append({
            'vehicle_id': vehicle_id,
            'action': 'return',
            'node_id': 0,
            'node_type': 'DEPOT',
            'start_time': current_time,
            'end_time': current_time + return_time,
            'tw': (Config.OPERATION_START, Config.OPERATION_END),
            'load_change': 0
        })

    return all_logs, len(sub_routes)


# ============================================================
# 甘特图绘制
# ============================================================

def plot_gantt(logs, num_orders, save_path, num_vehicles=None):
    """
    绘制甘特图

    参数:
    - logs: execution log 列表
    - num_orders: 订单数量 (25 / 50 / 100)
    - save_path: 图片保存路径
    - num_vehicles: 车辆数量 (可选，自动检测)
    """
    if plt is None or Rectangle is None or mpatches is None:
        raise RuntimeError("matplotlib 未安装，无法绘制甘特图；可先安装 matplotlib，或仅使用 --export_logs 导出日志。")

    if len(logs) == 0:
        print(f"警告: 没有日志数据，跳过 {save_path}")
        return

    # 检测车辆数量
    if num_vehicles is None:
        num_vehicles = max(log['vehicle_id'] for log in logs) + 1

    # 计算时间范围
    min_time = min(log['start_time'] for log in logs)
    max_time = max(log['end_time'] for log in logs)

    # 时间边界扩展
    time_padding = 0.5  # 30分钟
    x_min = max(Config.OPERATION_START - time_padding, min_time - time_padding)
    x_max = min(Config.OPERATION_END + 2, max_time + time_padding)

    # 图像尺寸 (根据车辆数自适应)
    fig_height = max(6, num_vehicles * 0.8 + 2)
    fig_width = 16

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 条形高度
    bar_height = 0.6

    # 绘制每个日志条目
    for log in logs:
        vehicle_id = log['vehicle_id']
        action = log['action']
        node_type = log['node_type']
        start = log['start_time']
        end = log['end_time']
        tw = log['tw']
        duration = end - start

        y_pos = num_vehicles - vehicle_id - 1  # 反转Y轴顺序

        # 选择颜色
        if action == 'move':
            color = COLORS['move']
            label_text = None
        elif action == 'wait':
            color = COLORS['wait']
            label_text = None
        elif action == 'serve':
            if node_type == 'P':
                color = COLORS['serve_pickup']
                label_text = f"P{log['node_id']}"
            else:
                color = COLORS['serve_delivery']
                label_text = f"D{log['node_id']}"
        elif action == 'return':
            color = COLORS['return']
            label_text = None
        else:
            color = '#808080'
            label_text = None

        # 检查是否迟到
        is_late = False
        if action == 'serve' and start > tw[1]:
            is_late = True

        # 绘制时间窗背景 (仅对 serve 行为)
        if action == 'serve':
            tw_rect = Rectangle(
                (tw[0], y_pos - bar_height/2 - 0.1),
                tw[1] - tw[0],
                bar_height + 0.2,
                facecolor=COLORS['tw_background'],
                edgecolor='none',
                alpha=0.4,
                zorder=1
            )
            ax.add_patch(tw_rect)

        # 绘制主条形
        if is_late:
            # 迟到: 红色边框 + 斜线填充
            rect = Rectangle(
                (start, y_pos - bar_height/2),
                duration,
                bar_height,
                facecolor=color,
                edgecolor=COLORS['late_border'],
                linewidth=2,
                hatch='///',
                zorder=3
            )
        else:
            rect = Rectangle(
                (start, y_pos - bar_height/2),
                duration,
                bar_height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                zorder=3
            )
        ax.add_patch(rect)

        # 添加标签文字 (仅对 serve)
        if label_text and duration > 0.1:
            text_x = start + duration / 2
            text_y = y_pos
            ax.text(text_x, text_y, label_text,
                    ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    color='white' if action == 'serve' else 'black',
                    zorder=4)

    # 设置坐标轴
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, num_vehicles - 0.5)

    # Y轴标签
    ax.set_yticks(range(num_vehicles))
    ax.set_yticklabels(
        [f'Vehicle {num_vehicles - i}' for i in range(num_vehicles)])

    # X轴格式化 (HH:MM)
    x_ticks = np.arange(int(x_min), int(x_max) + 1, 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([hours_to_hhmm(t)
                       for t in x_ticks], rotation=45, ha='right')

    # 仅显示X轴网格
    ax.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # 标签
    ax.set_xlabel('Time (HH:MM)', fontsize=12)
    ax.set_ylabel('Vehicle', fontsize=12)
    ax.set_title(f'Vehicle Scheduling Gantt Chart - {num_orders} Orders ({num_orders*2} Nodes)',
                 fontsize=14, fontweight='bold')

    # 创建图例
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['move'], edgecolor='black',
                       linewidth=0.5, label='Move'),
        mpatches.Patch(facecolor=COLORS['wait'], edgecolor='black',
                       linewidth=0.5, label='Wait'),
        mpatches.Patch(facecolor=COLORS['serve_pickup'], edgecolor='black',
                       linewidth=0.5, label='Pickup (P)'),
        mpatches.Patch(facecolor=COLORS['serve_delivery'], edgecolor='black',
                       linewidth=0.5, label='Delivery (D)'),
        mpatches.Patch(facecolor=COLORS['return'], edgecolor='black',
                       linewidth=0.5, label='Return'),
        mpatches.Patch(facecolor=COLORS['tw_background'], edgecolor='none',
                       alpha=0.4, label='Time Window'),
        mpatches.Patch(facecolor='white', edgecolor=COLORS['late_border'],
                       linewidth=2, hatch='///', label='Late Service Start')
    ]

    ax.legend(handles=legend_elements,
              loc='upper left',
              bbox_to_anchor=(1.01, 1),
              frameon=True,
              fancybox=True,
              shadow=True)

    # 布局优化
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"✓ 甘特图已保存: {save_path}")


# ============================================================
# 主程序
# ============================================================

def load_model_for_size(graph_size, device, model_path=None):
    """加载指定规模的训练模型（可显式传入 checkpoint 路径）"""

    if model_path is None:
        # 尝试优化版模型路径
        model_paths = [
            f'outputs/pomo_n{graph_size}_optimized/model_best.pt',
            f'outputs/pomo_n{graph_size}/model_best.pt',
            f'outputs/pomo_n{graph_size}_optimized/model_final.pt',
            f'outputs/pomo_n{graph_size}/model_final.pt'
        ]

        resolved_model_path = None
        for path in model_paths:
            if os.path.exists(path):
                resolved_model_path = path
                break

        if resolved_model_path is None:
            print(f"✗ 未找到 {graph_size} 订单的模型文件")
            return None
    else:
        resolved_model_path = model_path
        if not os.path.exists(resolved_model_path):
            print(f"✗ 指定模型文件不存在: {resolved_model_path}")
            return None

    print(f"  加载模型: {resolved_model_path}")
    checkpoint = torch.load(resolved_model_path, map_location=device)

    # 提取模型参数
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
        # 默认参数 (优化版)
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


def run_visualization():
    """主入口: 生成所有规模的甘特图"""

    print("=" * 70)
    print("MCVRP-PDTW 车辆排班甘特图生成器")
    print("Vehicle Scheduling Gantt Chart Generator")
    print("=" * 70)

    if plt is None:
        print("✗ matplotlib 未安装，run_visualization 模式无法绘制图片。")
        print("  可改用单规模模式并加 --export_logs 先导出执行日志。")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 创建输出目录
    output_dir = 'outputs/gantt_charts'
    os.makedirs(output_dir, exist_ok=True)

    # 问题规模列表
    problem_sizes = [25, 50, 100]

    for graph_size in problem_sizes:
        print(f"\n{'='*60}")
        print(f"处理 {graph_size} 订单问题 ({graph_size*2} 节点)")
        print("=" * 60)

        # 加载模型
        model = load_model_for_size(graph_size, device)
        if model is None:
            continue

        # 生成测试数据 (固定种子保证可复现)
        dataset = MCVRPPDTWDataset(
            num_samples=1,
            graph_size=graph_size,
            seed=2024  # 固定种子
        )
        batch = collate_fn([dataset[0]])

        # 生成执行日志
        print("  生成执行日志...")
        logs, num_vehicles = generate_execution_logs(model, batch, device)

        print(f"  车辆数量: {num_vehicles}")
        print(f"  日志条目: {len(logs)}")

        # 统计信息
        serve_logs = [l for l in logs if l['action'] == 'serve']
        pickup_logs = [l for l in serve_logs if l['node_type'] == 'P']
        delivery_logs = [l for l in serve_logs if l['node_type'] == 'D']
        late_logs = [l for l in serve_logs if l['start_time'] > l['tw'][1]]

        print(f"  取货服务: {len(pickup_logs)}")
        print(f"  送货服务: {len(delivery_logs)}")
        print(f"  迟到服务: {len(late_logs)}")

        # 绘制甘特图
        save_path = os.path.join(output_dir, f'gantt_{graph_size}.png')
        plot_gantt(logs, graph_size, save_path, num_vehicles)

    print(f"\n{'='*70}")
    print("所有甘特图生成完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 70)


def generate_single_gantt(graph_size, seed=2024, save_path=None):
    """
    生成单个规模的甘特图 (API接口)

    参数:
    - graph_size: 订单数量 (25/50/100)
    - seed: 随机种子
    - save_path: 保存路径 (可选)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model_for_size(graph_size, device)
    if model is None:
        return None, None

    dataset = MCVRPPDTWDataset(num_samples=1, graph_size=graph_size, seed=seed)
    batch = collate_fn([dataset[0]])

    logs, num_vehicles = generate_execution_logs(model, batch, device)

    if save_path is None:
        save_path = f'gantt_{graph_size}.png'

    plot_gantt(logs, graph_size, save_path, num_vehicles)

    return logs, num_vehicles


def export_logs(logs, export_path):
    """将执行日志导出为 CSV 或 JSON。"""
    if not export_path:
        return

    export_dir = os.path.dirname(export_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    ext = os.path.splitext(export_path)[1].lower()
    if ext == '.json':
        with open(export_path, 'w', encoding='utf-8') as fp:
            json.dump(logs, fp, ensure_ascii=False, indent=2)
        print(f"✓ 执行日志已导出(JSON): {export_path}")
        return

    if ext != '.csv':
        export_path = f"{export_path}.csv"

    fieldnames = [
        'vehicle_id', 'action', 'node_id', 'node_type',
        'start_time', 'end_time',
        'tw_start', 'tw_end',
        'load_change',
    ]
    with open(export_path, 'w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for log in logs:
            writer.writerow({
                'vehicle_id': log.get('vehicle_id'),
                'action': log.get('action'),
                'node_id': log.get('node_id'),
                'node_type': log.get('node_type'),
                'start_time': log.get('start_time'),
                'end_time': log.get('end_time'),
                'tw_start': (log.get('tw') or (None, None))[0],
                'tw_end': (log.get('tw') or (None, None))[1],
                'load_change': log.get('load_change'),
            })
    print(f"✓ 执行日志已导出(CSV): {export_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate vehicle gantt chart and optional execution logs')
    parser.add_argument('--graph_size', type=int, choices=[25, 50, 100, 200], default=None,
                        help='Generate only one graph size; omit to run default batch mode')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--save_path', type=str, default=None,
                        help='Output image path for single-size mode')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Optional checkpoint path override (supports n100/n200 custom path)')
    parser.add_argument('--export_logs', type=str, default=None,
                        help='Export execution logs to .csv/.json in single-size mode')
    return parser.parse_args()


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    args = parse_args()
    if args.graph_size is None:
        run_visualization()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model_for_size(args.graph_size, device, model_path=args.model_path)
        if model is None:
            raise SystemExit(1)
        dataset = MCVRPPDTWDataset(num_samples=1, graph_size=args.graph_size, seed=args.seed)
        batch = collate_fn([dataset[0]])
        logs, num_vehicles = generate_execution_logs(model, batch, device)
        export_logs(logs, args.export_logs)
        save_path = args.save_path or f'gantt_{args.graph_size}.png'
        if plt is None:
            print('! matplotlib 未安装，已导出执行日志，跳过甘特图绘制。')
        else:
            plot_gantt(logs, args.graph_size, save_path, num_vehicles)
