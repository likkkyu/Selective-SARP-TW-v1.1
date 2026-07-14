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
import math
import os
from datetime import datetime, timedelta

import numpy as np
import torch

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except Exception:  # pragma: no cover
    plt = None
    mpatches = None
    Rectangle = None

# 核心模块
from nets.attention_model import AttentionModel, set_decode_type
from problem_mcvrptw_v2 import Config, MCVRPPDTW, MCVRPPDTWDataset

# ============================================================
# 全局配置
# ============================================================

# 基准日期 (用于时间格式化)
BASE_DATE = datetime(2023, 1, 1)

SUPPORTED_THEMES = ('academic', 'classic')
SUPPORTED_LANGS = ('en', 'zh')


def get_gantt_style(theme='academic'):
    """返回甘特图绘图样式配置。"""
    theme = (theme or 'academic').lower()
    if theme not in SUPPORTED_THEMES:
        theme = 'academic'

    if theme == 'classic':
        return {
            'theme': 'classic',
            'colors': {
                'move': '#D3D3D3',
                'wait': '#FFD700',
                'serve_pickup': '#4169E1',
                'serve_delivery': '#32CD32',
                'return': '#696969',
                'tw_background': '#E6E6FA',
                'late_border': '#FF0000',
                'unknown': '#808080',
            },
            'rc': {
                'font.family': 'sans-serif',
                'font.sans-serif': ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans', 'Arial'],
                'font.size': 10,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'legend.fontsize': 9,
                'axes.linewidth': 1.0,
            },
            'bar_height': 0.60,
            'tw_pad': 0.10,
            'tw_alpha': 0.40,
            'tw_edgecolor': 'none',
            'bar_edgecolor': '#111111',
            'bar_linewidth': 0.55,
            'late_linewidth': 2.0,
            'late_hatch': '///',
            'major_grid_alpha': 0.70,
            'minor_grid_alpha': 0.0,
            'major_grid_color': '#8A8A8A',
            'minor_grid_color': '#D0D0D0',
            'show_minor_grid': False,
            'label_min_duration': 0.10,
            'label_fontsize': 7,
            'legend_framealpha': 1.0,
            'legend_borderpad': 0.6,
            'legend_handlelength': 1.6,
            'legend_fancybox': True,
            'legend_shadow': True,
            'title_weight': 'bold',
        }

    # academic / paper style
    return {
        'theme': 'academic',
        'colors': {
            'move': '#D9D9D9',
            'wait': '#F2C94C',
            'serve_pickup': '#3E6FB6',
            'serve_delivery': '#2FA84F',
            'return': '#595959',
            'tw_background': '#EEF2FF',
            'late_border': '#D62728',
            'unknown': '#8A8A8A',
        },
        'rc': {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Noto Sans CJK SC', 'SimHei'],
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 9,
            'axes.linewidth': 0.8,
            'xtick.labelsize': 9,
            'ytick.labelsize': 10,
        },
        'bar_height': 0.58,
        'tw_pad': 0.08,
        'tw_alpha': 0.28,
        'tw_edgecolor': '#DCE5FF',
        'bar_edgecolor': '#333333',
        'bar_linewidth': 0.45,
        'late_linewidth': 1.8,
        'late_hatch': '///',
        'major_grid_alpha': 0.32,
        'minor_grid_alpha': 0.18,
        'major_grid_color': '#9E9E9E',
        'minor_grid_color': '#C7C7C7',
        'show_minor_grid': True,
        'label_min_duration': 0.12,
        'label_fontsize': 7,
        'legend_framealpha': 0.95,
        'legend_borderpad': 0.45,
        'legend_handlelength': 1.5,
        'legend_fancybox': False,
        'legend_shadow': False,
        'title_weight': 'semibold',
    }


def get_text_bundle(lang='en'):
    """返回中英双语文案。"""
    lang = (lang or 'en').lower()
    if lang not in SUPPORTED_LANGS:
        lang = 'en'

    if lang == 'zh':
        return {
            'x_label': '时间 (HH:MM)',
            'y_label': '车辆',
            'vehicle_prefix': '车辆',
            'title': '车辆订单排班甘特图',
            'subtitle_suffix': '订单',
            'legend_move': '行驶 Move',
            'legend_wait': '等待 Wait',
            'legend_pickup': '接单 Pickup',
            'legend_delivery': '送达 Delivery',
            'legend_return': '回库 Return',
            'legend_tw': '时间窗 Time Window',
            'legend_late': '延迟送达 Late Service',
            'late_note': '延迟 {minutes:.0f} 分钟',
        }

    return {
        'x_label': 'Time (HH:MM)',
        'y_label': 'Vehicle',
        'vehicle_prefix': 'Vehicle',
        'title': 'Vehicle Scheduling Gantt Chart',
        'subtitle_suffix': 'orders',
        'legend_move': 'Move',
        'legend_wait': 'Wait',
        'legend_pickup': 'Pickup',
        'legend_delivery': 'Delivery',
        'legend_return': 'Return',
        'legend_tw': 'Time Window',
        'legend_late': 'Late Service',
        'late_note': 'Late {minutes:.0f} min',
    }


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


def _hex_to_rgb(hex_color):
    color = (hex_color or '#000000').lstrip('#')
    if len(color) != 6:
        return (0, 0, 0)
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _auto_text_color(hex_color):
    r, g, b = _hex_to_rgb(hex_color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return '#111111' if luminance >= 155 else '#FFFFFF'


def _service_label(log, n_orders):
    """将节点标签标准化为 Pk / Dk（k 为订单号，从 1 开始）。"""
    node_id = int(log.get('node_id', 0) or 0)
    node_type = log.get('node_type')

    if node_id <= 0:
        return None

    if node_type == 'P':
        order_id = node_id
        return f'P{order_id}'

    if node_type == 'D':
        order_id = node_id - int(n_orders)
        if order_id <= 0:
            order_id = node_id
        return f'D{order_id}'

    return None


def _parse_extra_formats(formats_str):
    if not formats_str:
        return []
    parsed = []
    for item in str(formats_str).split(','):
        fmt = item.strip().lower().lstrip('.')
        if fmt in {'png', 'pdf', 'svg'} and fmt not in parsed:
            parsed.append(fmt)
    return parsed


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
        _, _, pi = model(batch_input, return_pi=True)
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
        dist_to_first = np.linalg.norm(first_node_loc - depot) * Config.AREA_SIZE
        travel_time_to_first = dist_to_first / speed
        first_tw_start = tw[real_idx][0]
        trip_start_time = max(Config.OPERATION_START, first_tw_start - travel_time_to_first)

        current_time = trip_start_time

        for node_idx in sub_route:
            real_idx = node_idx - 1
            target_pos = locs[real_idx]

            # 计算行驶
            dist_km = np.linalg.norm(target_pos - current_pos) * Config.AREA_SIZE
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
                'load_change': 0,
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
                    'load_change': 0,
                })
                current_time = tw_start

            # 添加 serve 日志
            serve_start = current_time
            serve_end = current_time + service_time_hours

            # 计算载重变化
            if real_idx < n_orders:  # Pickup
                if node_type[real_idx] == 1:  # 乘客
                    load_change = int(round(dem_p[real_idx] * Config.PASSENGER_CAPACITY))
                else:  # 货物
                    load_change = int(round(dem_c[real_idx] * Config.CARGO_CAPACITY))
            else:  # Delivery
                if node_type[real_idx] == 1:
                    load_change = int(round(dem_p[real_idx] * Config.PASSENGER_CAPACITY))
                else:
                    load_change = int(round(dem_c[real_idx] * Config.CARGO_CAPACITY))

            all_logs.append({
                'vehicle_id': vehicle_id,
                'action': 'serve',
                'node_id': node_idx,
                'node_type': 'P' if real_idx < n_orders else 'D',
                'start_time': serve_start,
                'end_time': serve_end,
                'tw': (tw_start, tw_end),
                'load_change': load_change,
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
            'load_change': 0,
        })

    # 固定顺序，保证导出/绘图复现性
    all_logs.sort(key=lambda x: (int(x.get('vehicle_id', 0)), float(x.get('start_time', 0.0)), float(x.get('end_time', 0.0))))
    return all_logs, len(sub_routes)


# ============================================================
# 甘特图绘制
# ============================================================


def plot_gantt(
    logs,
    num_orders,
    save_path,
    num_vehicles=None,
    theme='academic',
    lang='en',
    dpi=300,
    annotate_late=False,
    late_topk=3,
    extra_formats=None,
):
    """
    绘制甘特图（支持学术风格、双语标签、额外导出格式）

    参数:
    - logs: execution log 列表
    - num_orders: 订单数量 (25 / 50 / 100 / 200)
    - save_path: 图片保存路径（主输出路径）
    - num_vehicles: 车辆数量 (可选，自动检测)
    - theme: academic | classic
    - lang: en | zh
    - dpi: PNG 输出分辨率
    - annotate_late: 是否标注延迟分钟
    - late_topk: 最多标注多少个最严重迟到
    - extra_formats: 额外输出格式列表，如 ['pdf', 'svg']
    """
    if plt is None or Rectangle is None or mpatches is None:
        raise RuntimeError('matplotlib 未安装，无法绘制甘特图；可先安装 matplotlib，或仅使用 --export_logs 导出日志。')

    if len(logs) == 0:
        print(f'警告: 没有日志数据，跳过 {save_path}')
        return []

    style = get_gantt_style(theme)
    texts = get_text_bundle(lang)
    colors = style['colors']

    # 检测车辆数量
    if num_vehicles is None:
        num_vehicles = max(log['vehicle_id'] for log in logs) + 1

    # 计算时间范围
    min_time = min(log['start_time'] for log in logs)
    max_time = max(log['end_time'] for log in logs)

    # 时间边界扩展
    time_padding = 0.45
    x_min = max(Config.OPERATION_START - time_padding, min_time - time_padding)
    x_max = min(Config.OPERATION_END + 2.0, max_time + time_padding)

    # 图像尺寸 (根据车辆数自适应)
    fig_height = max(6.0, num_vehicles * 0.75 + 2.1)
    fig_width = 16.2

    with plt.rc_context(style['rc']):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        bar_height = style['bar_height']

        late_candidates = []

        # 绘制每个日志条目
        for log in logs:
            vehicle_id = int(log['vehicle_id'])
            action = log['action']
            node_type = log['node_type']
            start = float(log['start_time'])
            end = float(log['end_time'])
            tw = log['tw']
            duration = max(0.0, end - start)

            y_pos = num_vehicles - vehicle_id - 1  # 反转Y轴顺序

            # 选择颜色
            if action == 'move':
                color = colors['move']
                label_text = None
            elif action == 'wait':
                color = colors['wait']
                label_text = None
            elif action == 'serve':
                color = colors['serve_pickup'] if node_type == 'P' else colors['serve_delivery']
                label_text = _service_label(log, num_orders)
            elif action == 'return':
                color = colors['return']
                label_text = None
            else:
                color = colors['unknown']
                label_text = None

            # 检查是否迟到
            is_late = bool(action == 'serve' and start > float(tw[1]) + 1e-9)

            # 绘制时间窗背景 (仅对 serve 行为)
            if action == 'serve':
                tw_start, tw_end = float(tw[0]), float(tw[1])
                tw_width = max(0.01, tw_end - tw_start)
                tw_rect = Rectangle(
                    (tw_start, y_pos - bar_height / 2 - style['tw_pad']),
                    tw_width,
                    bar_height + 2 * style['tw_pad'],
                    facecolor=colors['tw_background'],
                    edgecolor=style['tw_edgecolor'],
                    linewidth=0.4 if style['tw_edgecolor'] != 'none' else 0.0,
                    alpha=style['tw_alpha'],
                    zorder=1,
                )
                ax.add_patch(tw_rect)

            # 绘制主条形
            if is_late:
                rect = Rectangle(
                    (start, y_pos - bar_height / 2),
                    max(duration, 1e-6),
                    bar_height,
                    facecolor=color,
                    edgecolor=colors['late_border'],
                    linewidth=style['late_linewidth'],
                    hatch=style['late_hatch'],
                    zorder=3,
                )
                late_minutes = max(0.0, (start - float(tw[1])) * 60.0)
                late_candidates.append((late_minutes, y_pos, start, end, log))
            else:
                rect = Rectangle(
                    (start, y_pos - bar_height / 2),
                    max(duration, 1e-6),
                    bar_height,
                    facecolor=color,
                    edgecolor=style['bar_edgecolor'],
                    linewidth=style['bar_linewidth'],
                    zorder=3,
                )
            ax.add_patch(rect)

            # 标签文字 (仅对 serve，且条形足够长)
            if label_text and duration >= style['label_min_duration']:
                text_x = start + duration / 2
                text_y = y_pos
                text_color = _auto_text_color(color)
                ax.text(
                    text_x,
                    text_y,
                    label_text,
                    ha='center',
                    va='center',
                    fontsize=style['label_fontsize'],
                    fontweight='bold',
                    color=text_color,
                    zorder=4,
                )

        # 可选：标注 top-k 迟到分钟
        if annotate_late and late_candidates:
            late_candidates.sort(key=lambda item: item[0], reverse=True)
            for late_minutes, y_pos, start, end, _ in late_candidates[:max(0, int(late_topk))]:
                anchor_x = end + 0.06
                anchor_y = y_pos + 0.03
                note = texts['late_note'].format(minutes=late_minutes)
                ax.annotate(
                    note,
                    xy=(start, y_pos),
                    xytext=(anchor_x, anchor_y),
                    textcoords='data',
                    fontsize=8,
                    color=colors['late_border'],
                    arrowprops=dict(
                        arrowstyle='->',
                        color=colors['late_border'],
                        linewidth=0.8,
                        shrinkA=0,
                        shrinkB=3,
                    ),
                    zorder=5,
                )

        # 坐标轴设置
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, num_vehicles - 0.5)

        # Y 轴标签
        ax.set_yticks(range(num_vehicles))
        ax.set_yticklabels([f"{texts['vehicle_prefix']} {num_vehicles - i}" for i in range(num_vehicles)])

        # X 轴格式化 (HH:MM)
        major_left = int(math.floor(x_min))
        major_right = int(math.ceil(x_max))
        major_ticks = np.arange(major_left, major_right + 1, 1)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([hours_to_hhmm(t) for t in major_ticks], rotation=0)

        if style['show_minor_grid']:
            minor_ticks = np.arange(major_left, major_right + 0.5, 0.5)
            ax.set_xticks(minor_ticks, minor=True)

        # 仅显示 X 轴网格
        ax.grid(axis='x', which='major', linestyle='--', color=style['major_grid_color'], alpha=style['major_grid_alpha'], zorder=0)
        if style['show_minor_grid']:
            ax.grid(axis='x', which='minor', linestyle=':', color=style['minor_grid_color'], alpha=style['minor_grid_alpha'], zorder=0)
        ax.grid(axis='y', which='both', visible=False)
        ax.set_axisbelow(True)

        # 学术风格边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

        # 标签与标题
        ax.set_xlabel(texts['x_label'])
        ax.set_ylabel(texts['y_label'])
        ax.set_title(
            f"{texts['title']} ({num_orders} {texts['subtitle_suffix']})",
            fontweight=style['title_weight'],
        )

        # 图例
        legend_elements = [
            mpatches.Patch(facecolor=colors['move'], edgecolor=style['bar_edgecolor'], linewidth=style['bar_linewidth'], label=texts['legend_move']),
            mpatches.Patch(facecolor=colors['wait'], edgecolor=style['bar_edgecolor'], linewidth=style['bar_linewidth'], label=texts['legend_wait']),
            mpatches.Patch(facecolor=colors['serve_pickup'], edgecolor=style['bar_edgecolor'], linewidth=style['bar_linewidth'], label=texts['legend_pickup']),
            mpatches.Patch(facecolor=colors['serve_delivery'], edgecolor=style['bar_edgecolor'], linewidth=style['bar_linewidth'], label=texts['legend_delivery']),
            mpatches.Patch(facecolor=colors['return'], edgecolor=style['bar_edgecolor'], linewidth=style['bar_linewidth'], label=texts['legend_return']),
            mpatches.Patch(facecolor=colors['tw_background'], edgecolor=style['tw_edgecolor'], linewidth=0.5 if style['tw_edgecolor'] != 'none' else 0.0, alpha=style['tw_alpha'], label=texts['legend_tw']),
            mpatches.Patch(facecolor='white', edgecolor=colors['late_border'], linewidth=style['late_linewidth'], hatch=style['late_hatch'], label=texts['legend_late']),
        ]

        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            framealpha=style['legend_framealpha'],
            borderpad=style['legend_borderpad'],
            handlelength=style['legend_handlelength'],
            fancybox=style['legend_fancybox'],
            shadow=style['legend_shadow'],
        )

        # 布局优化
        plt.tight_layout()

        # 输出路径处理
        if not os.path.splitext(save_path)[1]:
            save_path = f'{save_path}.png'

        base, primary_ext = os.path.splitext(save_path)
        primary_ext = primary_ext.lower().lstrip('.')
        save_targets = [save_path]

        for fmt in extra_formats or []:
            fmt = str(fmt).lower().lstrip('.')
            if fmt in {'png', 'pdf', 'svg'}:
                target = f'{base}.{fmt}'
                if target not in save_targets:
                    save_targets.append(target)

        saved_paths = []
        for target in save_targets:
            ext = os.path.splitext(target)[1].lower()
            save_kwargs = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
            if ext == '.png':
                save_kwargs['dpi'] = int(dpi)
            fig.savefig(target, **save_kwargs)
            saved_paths.append(target)

        plt.close(fig)

    for path in saved_paths:
        print(f'✓ 甘特图已保存: {path}')
    return saved_paths


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
            f'outputs/pomo_n{graph_size}/model_final.pt',
        ]

        resolved_model_path = None
        for path in model_paths:
            if os.path.exists(path):
                resolved_model_path = path
                break

        if resolved_model_path is None:
            print(f'✗ 未找到 {graph_size} 订单的模型文件')
            return None
    else:
        resolved_model_path = model_path
        if not os.path.exists(resolved_model_path):
            print(f'✗ 指定模型文件不存在: {resolved_model_path}')
            return None

    print(f'  加载模型: {resolved_model_path}')
    checkpoint = torch.load(resolved_model_path, map_location=device)

    # 提取模型参数
    if 'args' in checkpoint and checkpoint['args'] is not None:
        args_dict = checkpoint['args']
        model_params = {
            'embedding_dim': args_dict.get('embedding_dim', 256),
            'hidden_dim': args_dict.get('hidden_dim', 256),
            'n_encode_layers': args_dict.get('n_encode_layers', 6),
            'tanh_clipping': args_dict.get('tanh_clipping', 10.0),
            'normalization': args_dict.get('normalization', 'batch'),
            'n_heads': args_dict.get('n_heads', 8),
            'problem': MCVRPPDTW,
        }
    else:
        # 默认参数 (优化版)
        model_params = {
            'embedding_dim': 256,
            'hidden_dim': 256,
            'n_encode_layers': 6,
            'tanh_clipping': 10.0,
            'normalization': 'batch',
            'n_heads': 8,
            'problem': MCVRPPDTW,
        }

    model = AttentionModel(**model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def run_visualization(theme='academic', lang='en', dpi=300, annotate_late=False, late_topk=3, extra_formats=None):
    """主入口: 生成所有规模的甘特图"""

    print('=' * 70)
    print('MCVRP-PDTW 车辆排班甘特图生成器')
    print('Vehicle Scheduling Gantt Chart Generator')
    print('=' * 70)

    if plt is None:
        print('✗ matplotlib 未安装，run_visualization 模式无法绘制图片。')
        print('  可改用单规模模式并加 --export_logs 先导出执行日志。')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 创建输出目录
    output_dir = 'outputs/gantt_charts'
    os.makedirs(output_dir, exist_ok=True)

    # 问题规模列表
    problem_sizes = [25, 50, 100]

    for graph_size in problem_sizes:
        print(f"\n{'=' * 60}")
        print(f'处理 {graph_size} 订单问题 ({graph_size * 2} 节点)')
        print('=' * 60)

        # 加载模型
        model = load_model_for_size(graph_size, device)
        if model is None:
            continue

        # 生成测试数据 (固定种子保证可复现)
        dataset = MCVRPPDTWDataset(
            num_samples=1,
            graph_size=graph_size,
            seed=2024,
        )
        batch = collate_fn([dataset[0]])

        # 生成执行日志
        print('  生成执行日志...')
        logs, num_vehicles = generate_execution_logs(model, batch, device)

        print(f'  车辆数量: {num_vehicles}')
        print(f'  日志条目: {len(logs)}')

        # 统计信息
        serve_logs = [l for l in logs if l['action'] == 'serve']
        pickup_logs = [l for l in serve_logs if l['node_type'] == 'P']
        delivery_logs = [l for l in serve_logs if l['node_type'] == 'D']
        late_logs = [l for l in serve_logs if l['start_time'] > l['tw'][1] + 1e-9]

        print(f'  取货服务: {len(pickup_logs)}')
        print(f'  送货服务: {len(delivery_logs)}')
        print(f'  迟到服务: {len(late_logs)}')

        # 绘制甘特图
        save_path = os.path.join(output_dir, f'gantt_{graph_size}.png')
        plot_gantt(
            logs,
            graph_size,
            save_path,
            num_vehicles,
            theme=theme,
            lang=lang,
            dpi=dpi,
            annotate_late=annotate_late,
            late_topk=late_topk,
            extra_formats=extra_formats,
        )

    print(f"\n{'=' * 70}")
    print('所有甘特图生成完成!')
    print(f'输出目录: {output_dir}')
    print('=' * 70)


def generate_single_gantt(
    graph_size,
    seed=2024,
    save_path=None,
    model_path=None,
    theme='academic',
    lang='en',
    dpi=300,
    annotate_late=False,
    late_topk=3,
    extra_formats=None,
):
    """
    生成单个规模的甘特图 (API接口)

    参数:
    - graph_size: 订单数量 (25/50/100/200)
    - seed: 随机种子
    - save_path: 保存路径 (可选)
    - model_path: 模型路径 (可选)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model_for_size(graph_size, device, model_path=model_path)
    if model is None:
        return None, None

    dataset = MCVRPPDTWDataset(num_samples=1, graph_size=graph_size, seed=seed)
    batch = collate_fn([dataset[0]])

    logs, num_vehicles = generate_execution_logs(model, batch, device)

    if save_path is None:
        save_path = f'gantt_{graph_size}.png'

    if plt is None:
        print('! matplotlib 未安装，跳过甘特图绘制。')
        return logs, num_vehicles

    plot_gantt(
        logs,
        graph_size,
        save_path,
        num_vehicles,
        theme=theme,
        lang=lang,
        dpi=dpi,
        annotate_late=annotate_late,
        late_topk=late_topk,
        extra_formats=extra_formats,
    )

    return logs, num_vehicles


def export_logs(logs, export_path):
    """将执行日志导出为 CSV 或 JSON。"""
    if not export_path:
        return

    export_dir = os.path.dirname(export_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    logs_sorted = sorted(
        logs,
        key=lambda x: (
            int(x.get('vehicle_id', 0)),
            float(x.get('start_time', 0.0)),
            float(x.get('end_time', 0.0)),
        ),
    )

    ext = os.path.splitext(export_path)[1].lower()
    if ext == '.json':
        with open(export_path, 'w', encoding='utf-8') as fp:
            json.dump(logs_sorted, fp, ensure_ascii=False, indent=2)
        print(f'✓ 执行日志已导出(JSON): {export_path}')
        return

    if ext != '.csv':
        export_path = f'{export_path}.csv'

    fieldnames = [
        'vehicle_id',
        'action',
        'node_id',
        'node_type',
        'start_time',
        'end_time',
        'tw_start',
        'tw_end',
        'load_change',
    ]
    with open(export_path, 'w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for log in logs_sorted:
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
    print(f'✓ 执行日志已导出(CSV): {export_path}')


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

    # 新增：学术风格可视化参数
    parser.add_argument('--theme', choices=list(SUPPORTED_THEMES), default='academic',
                        help='Plot theme style (default: academic)')
    parser.add_argument('--lang', choices=list(SUPPORTED_LANGS), default='en',
                        help='Language for title/labels (default: en)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='PNG export DPI (default: 300)')
    parser.add_argument('--formats', type=str, default='',
                        help='Additional figure formats, comma-separated: pdf,svg,png')
    parser.add_argument('--annotate_late', action='store_true', default=False,
                        help='Annotate top-k late service blocks on chart')
    parser.add_argument('--late_topk', type=int, default=3,
                        help='Top-k late blocks to annotate when --annotate_late is enabled')
    parser.add_argument('--no_plot', action='store_true', default=False,
                        help='Skip plotting and only export logs')
    return parser.parse_args()


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    args = parse_args()
    extra_formats = _parse_extra_formats(args.formats)

    if args.graph_size is None:
        if args.no_plot:
            print('! --no_plot 在批量模式下将跳过全部绘图。可改用单规模模式配合 --export_logs。')
        run_visualization(
            theme=args.theme,
            lang=args.lang,
            dpi=args.dpi,
            annotate_late=args.annotate_late,
            late_topk=args.late_topk,
            extra_formats=extra_formats,
        )
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model_for_size(args.graph_size, device, model_path=args.model_path)
        if model is None:
            raise SystemExit(1)

        dataset = MCVRPPDTWDataset(num_samples=1, graph_size=args.graph_size, seed=args.seed)
        batch = collate_fn([dataset[0]])
        logs, num_vehicles = generate_execution_logs(model, batch, device)

        export_logs(logs, args.export_logs)

        if args.no_plot:
            print('! 已按 --no_plot 跳过甘特图绘制。')
            raise SystemExit(0)

        save_path = args.save_path or f'gantt_{args.graph_size}.png'
        if plt is None:
            print('! matplotlib 未安装，已导出执行日志，跳过甘特图绘制。')
        else:
            plot_gantt(
                logs,
                args.graph_size,
                save_path,
                num_vehicles,
                theme=args.theme,
                lang=args.lang,
                dpi=args.dpi,
                annotate_late=args.annotate_late,
                late_topk=args.late_topk,
                extra_formats=extra_formats,
            )
