"""
Selective SARP-TW v1.1 的客货混合 Pickup-Delivery 车辆路径问题主实现。
Multi-Compartment Vehicle Routing Problem with Pickup-Delivery and Time Windows

当前主路径语义（2026-06）：
1. 业务定位：响应式低峰客货共享调度，static-batch（day-ahead）规划
2. 车队 K_max = ceil(N/6) 硬上限（当前代码口径，后续若重设需同步文档）
3. 货物语义：每单 1-3 单位（能耗按每单位 1 kg 计），舱容 20 单位
4. 乘客语义：每单 1-4 人（80% 为 {1,2}，20% 为 {3,4}），舱容 15 人
5. 类型分布：乘客 60% / 货物 40%
6. passenger pickup：硬时间窗
7. passenger maximum ride time：总时长 70 min / 超额 30 min 硬约束
8. passenger delivery：软迟到成本；cargo 继续仅在 delivery 端统计软迟到
9. 16:00 运营硬约束：DRL mask 禁止任何会超出 16:00 的非-depot 节点
10. 单趟硬上限：3h（DRL mask + 软兜底 ALPHA_TRIP_OVERTIME=200）
11. 训练目标 = objective_total（归一化+α 加权+主动 reject + 未履约兜底）
    报告目标 = total_cost_raw（纯 RMB，三 baseline 同尺度）

成本函数（论文/评测口径，raw RMB）：
    Z = c_elec·ΣW_ij + ω_p·ΣDelay_i^{p,delivery} + ω_c·ΣDelay_i^{c,delivery}
        + ω_v·K_used + reject_penalty + unfulfilled_penalty + trip_overtime_penalty
其中：
    - c_elec = 1.0 元/kWh
    - ω_p = 0.6 元/min（仅乘客 delivery）
    - ω_c = 0.06 元/min（仅货物 delivery）
    - ω_v = 20 元/车
    - reject = 500 元/主动 reject 订单
    - unfulfilled = 600 元/未显式 reject 但最终未完成订单
    - trip_overtime = 200 元/h（safety net，合理解里 ≈ 0）
"""

import json
import math
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans


# ============ 全局参数配置 ============
class Config:
    """问题与成本计算的唯一参数源（Selective SARP-TW v1.1）。"""

    # ---------- 空间范围 ----------
    AREA_SIZE = 10.0  # 10 km × 10 km

    # ---------- 容量限制（v6: 货物单位改为 kg） ----------
    PASSENGER_CAPACITY = 15   # 乘客容量（人）
    CARGO_CAPACITY = 20       # 货物容量（单位）

    # ---------- 时间参数 ----------
    OPERATION_START = 10.0   # 10:00
    OPERATION_END = 16.0     # 16:00
    MAX_TRIP_TIME = 3.0      # 单趟最长 3h
    PASSENGER_TW_WIDTH = 1.0 # 乘客 TW 宽度（1 h）
    CARGO_TW_WIDTH = 1.0     # 货物 TW 宽度（1h）
    SERVICE_TIME = 3.0 / 60  # 3 min

    # ---------- 车辆参数 ----------
    VEHICLE_SPEED = 25.0     # km/h
    # 能耗系数采用 MODELING.md §6 公式：η(W) = ENERGY_BASE · (1 + W / ENERGY_WEIGHT_REF)
    # 其中 W 为该弧上车辆实时载重 (kg)。空载 η = 0.18 kWh/km。
    ENERGY_BASE = 0.18           # 空载基础能耗 (kWh/km)
    ENERGY_WEIGHT_REF = 10000.0  # 载重缩放参考 (kg)

    # ---------- 成本参数（元） ----------
    ELECTRICITY_PRICE = 1.0      # 元/kWh
    PASSENGER_DELAY_COST = 0.6   # ω_p, 元/min（仅 delivery）
    CARGO_DELAY_COST = 0.06      # ω_c, 元/min（仅 delivery）
    VEHICLE_COST = 20.0          # ω_v, 元/车·班次（更贴近静态多车日计划中的固定派车成本）

    # ---------- α 加权（仅训练目标使用，论文公式不出现） ----------
    ALPHA_ENERGY = 1.0
    ALPHA_DELAY = 2.5
    ALPHA_VEHICLE = 3.0
    ALPHA_REJECT = 500.0          # 元/主动 reject 订单
    ALPHA_UNFULFILLED = 600.0     # 元/未显式 reject 但最终未完成订单（高于 reject，避免静默漏单）
    ALPHA_TRIP_OVERTIME = 200.0   # 元/h 单趟超时（safety net）

    # ---------- 约束开关（服务质量强化版） ----------
    HARD_PASSENGER_PICKUP_TIMEWINDOW = True   # passenger pickup 硬时间窗
    HARD_PASSENGER_MAX_RIDE_TIME = True       # passenger maximum ride time 双层硬约束
    PASSENGER_MAX_RIDE_TIME_MINUTES = 70.0    # passenger total ride time: 70 min
    PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES = 30.0  # passenger excess ride time: 30 min
    HARD_MAX_TRIP_TIME = True                 # 单趟 3h 硬约束
    HARD_OPERATION_END = True                 # 16:00 硬约束
    HARD_VEHICLE_LIMIT = True                 # K_max 硬上限

    # ---------- 物理常量（能耗模型用） ----------
    PASSENGER_WEIGHT_KG = 65.0   # 单人重量（kg）
    CARGO_UNIT_WEIGHT_KG = 1.0   # v6: 单位货物重量 1 kg（外卖/快递包裹）

    # ---------- 数据生成参数（v1.1: 乘客 60% / 货物 40%） ----------
    PASSENGER_RATIO = 0.6
    PASSENGER_RATIO_LARGE = 0.6
    RANDOM_RATIO = 0.5
    CLUSTER_RATIO = 0.5
    NUM_CLUSTERS = 3
    PD_SHORT_DISTANCE_RATIO = 0.8   # 80% cargo 订单起终点距离不超过 10 km
    PD_SHORT_DISTANCE_KM = 10.0
    PD_LONG_DISTANCE_KM = 15.0      # 剩余 cargo 订单起终点距离不超过 15 km
    PASSENGER_PD_LONG_RATIO = 0.05
    PASSENGER_PD_MID_RATIO = 0.35
    PASSENGER_PD_SHORT_RATIO = 0.60
    PASSENGER_PD_LONG_MIN_KM = 10.0
    PASSENGER_PD_LONG_MAX_KM = 15.0
    PASSENGER_PD_MID_MIN_KM = 5.0
    PASSENGER_PD_MID_MAX_KM = 10.0
    PASSENGER_PD_SHORT_MIN_KM = 3.0
    PASSENGER_PD_SHORT_MAX_KM = 5.0

    DEMAND_MIN = 1   # 单订单最小 demand（乘客=人，货物=kg）
    DEMAND_MAX = 3   # 货物需求上限（乘客大组由后续采样覆盖到 4）
    PASSENGER_SMALL_GROUP_RATIO = 0.8  # 80% 乘客订单为 1-2 人

    # ---------- 时间窗分布参数（按时段混合采样） ----------
    # 三个时段：[10,12), [12,14), [14,16)
    PASSENGER_TW_PERIOD_WEIGHTS = (0.50, 0.32, 0.18)
    CARGO_TW_PERIOD_WEIGHTS = (0.52, 0.30, 0.18)

    # ---------- 车队规模（50 单主线训练回到较宽松预算，先学服务，再收紧） ----------
    MIN_NUM_VEHICLES = 1
    DEFAULT_NUM_VEHICLE_RATIO = 1.0 / 6.0

    # ---------- 归一化参数 ----------
    NORMALIZATION_OUTPUT_DIR = 'outputs/normalization_profiles'
    NORMALIZATION_PROFILES = {
        25: {
            'graph_size': 25,
            'energy_cost_raw': 1.0,
            'passenger_delivery_delay_cost_raw': 1.0,
            'cargo_delay_cost_raw': 1.0,
            'vehicle_cost_norm': 1.0,
            'num_samples': 0,
            'seed': None,
            'num_vehicles': None
        },
        50: {
            'graph_size': 50,
            'energy_cost_raw': 1.0,
            'passenger_delivery_delay_cost_raw': 1.0,
            'cargo_delay_cost_raw': 1.0,
            'vehicle_cost_norm': 1.0,
            'num_samples': 0,
            'seed': None,
            'num_vehicles': None
        },
        100: {
            'graph_size': 100,
            'energy_cost_raw': 1.0,
            'passenger_delivery_delay_cost_raw': 1.0,
            'cargo_delay_cost_raw': 1.0,
            'vehicle_cost_norm': 1.0,
            'num_samples': 0,
            'seed': None,
            'num_vehicles': None
        }
    }

    @classmethod
    def get_default_num_vehicles(cls, n_orders):
        """当前默认 K_max = ceil(N/6) 硬上限。"""
        return max(cls.MIN_NUM_VEHICLES, int(math.ceil(n_orders * cls.DEFAULT_NUM_VEHICLE_RATIO)))

    @classmethod
    def get_normalization_output_path(cls, graph_size, base_dir=None):
        root_dir = base_dir or cls.NORMALIZATION_OUTPUT_DIR
        return os.path.join(root_dir, f'normalization_n{graph_size}.json')

    @classmethod
    def _default_normalization_profile(cls, graph_size):
        return {
            'graph_size': int(graph_size),
            'energy_cost_raw': 1.0,
            'passenger_delivery_delay_cost_raw': 1.0,
            'cargo_delay_cost_raw': 1.0,
            'vehicle_cost_norm': 1.0,
            'num_samples': 0,
            'seed': None,
            'num_vehicles': None
        }

    @classmethod
    def set_normalization_profile(cls, graph_size, profile):
        """更新归一化配置，并强制下界>=1.0，避免数值爆炸 (F5 bug fix)。"""
        merged_profile = cls._default_normalization_profile(graph_size)
        merged_profile.update(profile or {})
        # 下界保护：任何 norm 常数都不能小于 1.0
        for key in ('energy_cost_raw', 'passenger_delivery_delay_cost_raw', 'cargo_delay_cost_raw', 'vehicle_cost_norm'):
            merged_profile[key] = max(float(merged_profile.get(key, 1.0)), 1.0)
        cls.NORMALIZATION_PROFILES[int(graph_size)] = merged_profile
        return merged_profile

    @classmethod
    def load_normalization_profile(cls, graph_size, base_dir=None):
        graph_size = int(graph_size)
        profile_path = cls.get_normalization_output_path(graph_size, base_dir=base_dir)
        if os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as profile_file:
                import json
                profile_data = json.load(profile_file)
                if 'passenger_delivery_delay_cost_raw' not in profile_data and 'passenger_penalty_raw' in profile_data:
                    profile_data['passenger_delivery_delay_cost_raw'] = profile_data['passenger_penalty_raw']
                if 'vehicle_cost_norm' not in profile_data:
                    print(f"[Warning] Outdated normalization profile found for N={graph_size} (missing vehicle_cost_norm). Re-calibrating...")
                    calibrate_normalization(graph_size, output_path=profile_path)
                    with open(profile_path, 'r', encoding='utf-8') as new_profile_file:
                        profile_data = json.load(new_profile_file)
                        if 'passenger_delivery_delay_cost_raw' not in profile_data and 'passenger_penalty_raw' in profile_data:
                            profile_data['passenger_delivery_delay_cost_raw'] = profile_data['passenger_penalty_raw']
                return cls.set_normalization_profile(graph_size, profile_data)
        
        # If it doesn't exist, we fallback to default but maybe we should calibrate it right away.
        print(f"[Warning] Normalization profile not found for N={graph_size}. Falling back to default.")
        return cls.NORMALIZATION_PROFILES.get(graph_size, cls._default_normalization_profile(graph_size))

    @classmethod
    def get_normalization_profile(cls, graph_size):
        graph_size = int(graph_size)
        if graph_size not in cls.NORMALIZATION_PROFILES:
            return cls.load_normalization_profile(graph_size)
        return cls.NORMALIZATION_PROFILES[graph_size]


class MCVRPPDTW:
    """
    客货混合Pickup-Delivery VRP with Time Windows问题类
    
    问题结构：
    - 节点0：depot（车场）
    - 节点1到n/2：pickup点（接客/货点）
    - 节点n/2+1到n：delivery点（送达点）
    - pickup节点i对应delivery节点i+n/2
    """
    
    NAME = 'mcvrppdtw'  # Multi-Compartment VRP with Pickup-Delivery and Time Windows
    
    # 容量上限（归一化后的值）
    PASSENGER_CAPACITY = 1.0
    CARGO_CAPACITY = 1.0
    
    @staticmethod
    def _compute_distance_energy(dataset, pi, loc_with_depot):
        """[v5 重构] 计算每段距离与能耗。

        关键修复: cumulative_demand 在 pi==0 (车辆切换) 时强制清零，与 state_mcvrptw_v2.update 的语义保持一致。
        """
        batch_size = pi.size(0)
        seq_len = pi.size(1)
        device = pi.device

        coords_in_order = loc_with_depot.gather(1, pi[..., None].expand(-1, -1, 2))
        coords_km = coords_in_order * Config.AREA_SIZE
        segment_distances = (coords_km[:, 1:] - coords_km[:, :-1]).norm(p=2, dim=2)
        total_distance = segment_distances.sum(1)

        demand_p_with_depot = torch.cat(
            (torch.zeros(batch_size, 1, device=device), dataset['demand_passenger']), dim=1
        )
        demand_c_with_depot = torch.cat(
            (torch.zeros(batch_size, 1, device=device), dataset['demand_cargo']), dim=1
        )
        demand_p_in_order = demand_p_with_depot.gather(1, pi)
        demand_c_in_order = demand_c_with_depot.gather(1, pi)

        cumulative_demand_p = torch.zeros(batch_size, 1, device=device)
        cumulative_demand_c = torch.zeros(batch_size, 1, device=device)
        segment_weights_kg = []

        for t in range(seq_len - 1):
            # v6: 货物单位改为 1 kg / 单位（外卖/快递包裹）
            weight_p = cumulative_demand_p * Config.PASSENGER_CAPACITY * Config.PASSENGER_WEIGHT_KG
            weight_c = cumulative_demand_c * Config.CARGO_CAPACITY * Config.CARGO_UNIT_WEIGHT_KG
            total_weight_kg = weight_p + weight_c
            segment_weights_kg.append(total_weight_kg)

            # 累加当前节点的需求 (delivery 为负，会自动抵扣)
            cumulative_demand_p = cumulative_demand_p + demand_p_in_order[:, t:t + 1]
            cumulative_demand_c = cumulative_demand_c + demand_c_in_order[:, t:t + 1]
            cumulative_demand_p = torch.clamp(cumulative_demand_p, min=0.0)
            cumulative_demand_c = torch.clamp(cumulative_demand_c, min=0.0)

            # ★ v5 关键修复：若下一步是 depot（pi==0），代表车辆切换，强制清零累计载量
            #   原版本依赖 pickup/delivery 自然抵消，违规解会把残余载量带给下一辆车，造成能耗计费失真
            next_node = pi[:, t + 1:t + 2]
            is_next_depot = (next_node == 0).float()
            cumulative_demand_p = cumulative_demand_p * (1.0 - is_next_depot)
            cumulative_demand_c = cumulative_demand_c * (1.0 - is_next_depot)

        if segment_weights_kg:
            segment_weights_kg = torch.cat(segment_weights_kg, dim=1)
        else:
            segment_weights_kg = torch.zeros(batch_size, 0, device=device)

        # MODELING.md §6: η(W) = ENERGY_BASE * (1 + W / ENERGY_WEIGHT_REF) [kWh/km]
        # W 为该弧上车辆实时载重 (kg)。空载 W=0 时 η = ENERGY_BASE = 0.18 kWh/km。
        dynamic_energy_rate = Config.ENERGY_BASE * (1.0 + segment_weights_kg / Config.ENERGY_WEIGHT_REF)
        segment_energy = dynamic_energy_rate * segment_distances
        energy_cost_raw = segment_energy.sum(1) * Config.ELECTRICITY_PRICE
        return energy_cost_raw, segment_distances, total_distance

    @staticmethod
    def _compute_time_and_delay(pi, segment_distances, tw_in_order, type_in_order, n_orders, dataset):
        """统一时间仿真。

        当前服务质量语义：
        1. passenger pickup 为硬时间窗（主路径由 state.get_mask 保证）
        2. passenger maximum ride time 为总时长 70 min / 超额 30 min 硬约束（主路径由 state.get_mask 保证）
        3. passenger delivery 保持软迟到成本
        4. cargo 继续仅在 delivery 节点统计软迟到
        5. 早到等待，不罚成本；当前不显式建模 depot 端延迟出发决策
        """
        batch_size, seq_len = pi.size()
        device = pi.device
        loc_from_dataset = dataset['loc']

        travel_times = segment_distances / Config.VEHICLE_SPEED
        current_time = torch.full((batch_size,), Config.OPERATION_START, device=device)
        passenger_delivery_delay_minutes = torch.zeros(batch_size, device=device)
        cargo_delay_minutes = torch.zeros(batch_size, device=device)
        passenger_pickup_violations = torch.zeros(batch_size, device=device)
        passenger_total_ride_time_violations = torch.zeros(batch_size, device=device)
        passenger_excess_ride_time_violations = torch.zeros(batch_size, device=device)
        passenger_ride_time_minutes_total = torch.zeros(batch_size, device=device)
        passenger_ride_time_minutes_max = torch.zeros(batch_size, device=device)
        passenger_excess_ride_time_minutes_total = torch.zeros(batch_size, device=device)
        passenger_excess_ride_time_minutes_max = torch.zeros(batch_size, device=device)
        passenger_pickup_total = torch.zeros(batch_size, device=device)
        passenger_pickup_on_time = torch.zeros(batch_size, device=device)
        passenger_delivery_total = torch.zeros(batch_size, device=device)
        passenger_delivery_on_time = torch.zeros(batch_size, device=device)
        cargo_delivery_total = torch.zeros(batch_size, device=device)
        cargo_delivery_on_time = torch.zeros(batch_size, device=device)
        passenger_pickup_finish_time = torch.full((batch_size, n_orders), -1.0, device=device)

        arrival_times = torch.zeros(batch_size, seq_len, device=device)
        delay_hours = torch.zeros(batch_size, seq_len, device=device)
        overtime_hours = torch.zeros(batch_size, seq_len, device=device)
        trip_overtime_hours = torch.zeros(batch_size, seq_len, device=device)

        if n_orders > 0:
            direct_distance = torch.norm(
                loc_from_dataset[:, :n_orders, :] - loc_from_dataset[:, n_orders:2 * n_orders, :],
                dim=-1,
            ) * Config.AREA_SIZE
            direct_ride_time_minutes_all = direct_distance / Config.VEHICLE_SPEED * 60
        else:
            direct_ride_time_minutes_all = torch.zeros(batch_size, 0, device=device)

        trip_start_time = torch.full((batch_size,), Config.OPERATION_START, device=device)
        depot_node = 0

        for t in range(seq_len):
            if t > 0:
                current_time = current_time + travel_times[:, t - 1]

            node_idx = pi[:, t]
            is_depot = (node_idx == depot_node).float()

            if t > 0:
                trip_duration = current_time - trip_start_time
                trip_over = torch.clamp(trip_duration - Config.MAX_TRIP_TIME, min=0.0)
                trip_overtime_hours[:, t] = is_depot * trip_over
                current_time = is_depot * Config.OPERATION_START + (1 - is_depot) * current_time
                trip_start_time = is_depot * Config.OPERATION_START + (1 - is_depot) * trip_start_time

            arrival_times[:, t] = current_time

            latest = tw_in_order[:, t, 1]
            delay = torch.clamp(current_time - latest, min=0.0)
            delay_hours[:, t] = delay
            delay_minutes = delay * 60

            is_pickup = (node_idx >= 1) & (node_idx <= n_orders)
            is_delivery = (node_idx >= n_orders + 1) & (node_idx <= 2 * n_orders)
            is_passenger = type_in_order[:, t] == 1
            is_cargo = type_in_order[:, t] == 0

            passenger_pickup_mask = is_pickup & is_passenger
            passenger_delivery_mask = is_delivery & is_passenger
            cargo_delivery_mask = is_delivery & is_cargo

            passenger_pickup_violations = passenger_pickup_violations + (passenger_pickup_mask & (delay > 1e-5)).float()
            passenger_pickup_total = passenger_pickup_total + passenger_pickup_mask.float()
            passenger_pickup_on_time = passenger_pickup_on_time + (passenger_pickup_mask & (delay <= 1e-5)).float()
            passenger_delivery_total = passenger_delivery_total + passenger_delivery_mask.float()
            passenger_delivery_on_time = passenger_delivery_on_time + (passenger_delivery_mask & (delay <= 1e-5)).float()
            cargo_delivery_total = cargo_delivery_total + cargo_delivery_mask.float()
            cargo_delivery_on_time = cargo_delivery_on_time + (cargo_delivery_mask & (delay <= 1e-5)).float()

            passenger_delivery_delay_minutes = passenger_delivery_delay_minutes + (passenger_delivery_mask.float() * delay_minutes)
            cargo_delay_minutes = cargo_delay_minutes + (cargo_delivery_mask.float() * delay_minutes)

            earliest = tw_in_order[:, t, 0]
            start_service = torch.max(current_time, earliest)
            current_time = start_service + Config.SERVICE_TIME

            if n_orders > 0:
                pickup_order_idx = (node_idx - 1).clamp(min=0, max=n_orders - 1)
                delivery_order_idx = (node_idx - (n_orders + 1)).clamp(min=0, max=n_orders - 1)

                if passenger_pickup_mask.any():
                    pickup_rows = torch.nonzero(passenger_pickup_mask, as_tuple=False).squeeze(-1)
                    passenger_pickup_finish_time[pickup_rows, pickup_order_idx[pickup_rows]] = current_time[pickup_rows]

                delivery_pickup_finish = passenger_pickup_finish_time.gather(1, delivery_order_idx[:, None]).squeeze(1)
                valid_passenger_delivery = passenger_delivery_mask & (delivery_pickup_finish >= 0)
                ride_time_minutes = torch.clamp((arrival_times[:, t] - delivery_pickup_finish) * 60, min=0.0)
                direct_ride_time_minutes = direct_ride_time_minutes_all.gather(1, delivery_order_idx[:, None]).squeeze(1)
                excess_ride_time_minutes = torch.clamp(ride_time_minutes - direct_ride_time_minutes, min=0.0)

                valid_delivery_float = valid_passenger_delivery.float()
                passenger_ride_time_minutes_total = passenger_ride_time_minutes_total + valid_delivery_float * ride_time_minutes
                passenger_ride_time_minutes_max = torch.maximum(
                    passenger_ride_time_minutes_max,
                    torch.where(valid_passenger_delivery, ride_time_minutes, torch.zeros_like(ride_time_minutes)),
                )
                passenger_excess_ride_time_minutes_total = passenger_excess_ride_time_minutes_total + valid_delivery_float * excess_ride_time_minutes
                passenger_excess_ride_time_minutes_max = torch.maximum(
                    passenger_excess_ride_time_minutes_max,
                    torch.where(valid_passenger_delivery, excess_ride_time_minutes, torch.zeros_like(excess_ride_time_minutes)),
                )
                passenger_total_ride_time_violations = passenger_total_ride_time_violations + (
                    valid_passenger_delivery & (ride_time_minutes > Config.PASSENGER_MAX_RIDE_TIME_MINUTES + 1e-5)
                ).float()
                passenger_excess_ride_time_violations = passenger_excess_ride_time_violations + (
                    valid_passenger_delivery & (excess_ride_time_minutes > Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES + 1e-5)
                ).float()

            overtime = torch.clamp(current_time - Config.OPERATION_END, min=0.0)
            overtime_hours[:, t] = overtime

        last_trip_duration = current_time - trip_start_time
        last_trip_over = torch.clamp(last_trip_duration - Config.MAX_TRIP_TIME, min=0.0)
        trip_overtime_hours[:, -1] = trip_overtime_hours[:, -1] + last_trip_over

        return {
            'arrival_times': arrival_times,
            'delay_hours': delay_hours,
            'overtime_hours': overtime_hours,
            'trip_overtime_hours': trip_overtime_hours,
            'passenger_delivery_delay_minutes': passenger_delivery_delay_minutes,
            'cargo_delay_minutes': cargo_delay_minutes,
            'passenger_pickup_violations': passenger_pickup_violations,
            'passenger_ride_time_violations': passenger_total_ride_time_violations,
            'passenger_total_ride_time_violations': passenger_total_ride_time_violations,
            'passenger_excess_ride_time_violations': passenger_excess_ride_time_violations,
            'passenger_ride_time_minutes_total': passenger_ride_time_minutes_total,
            'passenger_ride_time_minutes_max': passenger_ride_time_minutes_max,
            'passenger_excess_ride_time_minutes_total': passenger_excess_ride_time_minutes_total,
            'passenger_excess_ride_time_minutes_max': passenger_excess_ride_time_minutes_max,
            'passenger_pickup_total': passenger_pickup_total,
            'passenger_pickup_on_time': passenger_pickup_on_time,
            'passenger_delivery_total': passenger_delivery_total,
            'passenger_delivery_on_time': passenger_delivery_on_time,
            'cargo_delivery_total': cargo_delivery_total,
            'cargo_delivery_on_time': cargo_delivery_on_time,
            'type_in_order': type_in_order,
            'final_time': current_time,
        }

    @staticmethod
    def _compute_vehicle_and_penalty(pi, graph_size, time_dict):
        """[v6 重构] 车辆数 / 拒单 / 单趟超时 软惩罚集中计算。"""
        batch_size = pi.size(0)
        device = pi.device

        pickup_mask = (pi >= 1) & (pi <= graph_size)

        depot_node = 0
        prev_is_depot = torch.cat(
            (
                torch.ones(batch_size, 1, device=device, dtype=torch.bool),
                (pi[:, :-1] == depot_node),
            ),
            dim=1,
        )
        vehicle_starts = pickup_mask & prev_is_depot
        used_vehicles = torch.clamp(vehicle_starts.sum(1).float(), min=1.0)

        vehicle_cost_raw = used_vehicles * Config.VEHICLE_COST

        order_idx = torch.arange(graph_size, device=device)
        pickup_visit = (pi[:, :, None] == (order_idx + 1).view(1, 1, -1)).any(dim=1)
        delivery_visit = (pi[:, :, None] == (order_idx + graph_size + 1).view(1, 1, -1)).any(dim=1)
        completed_orders_mask = pickup_visit & delivery_visit
        completed_orders = completed_orders_mask.float().sum(1)
        unserved_orders = torch.clamp(graph_size - completed_orders, min=0.0)

        trip_overtime_total = time_dict['trip_overtime_hours'].sum(1)
        trip_overtime_penalty = trip_overtime_total * Config.ALPHA_TRIP_OVERTIME

        reject_count = time_dict.get('reject_count', torch.zeros_like(unserved_orders))
        explicit_reject_mask = (~pickup_visit) & (~delivery_visit)
        rejected_orders = explicit_reject_mask.float().sum(1).clamp(max=reject_count)
        pickup_only_mask = pickup_visit & (~delivery_visit)
        delivery_without_pickup_mask = delivery_visit & (~pickup_visit)
        started_not_completed_mask = pickup_only_mask | delivery_without_pickup_mask
        pickup_only_orders = pickup_only_mask.float().sum(1)
        delivery_without_pickup_orders = delivery_without_pickup_mask.float().sum(1)
        started_not_completed_orders = started_not_completed_mask.float().sum(1)
        active_rejected_orders = torch.minimum(reject_count, unserved_orders)
        unfulfilled_orders = torch.clamp(unserved_orders - active_rejected_orders, min=0.0)
        reject_penalty = active_rejected_orders * Config.ALPHA_REJECT
        unfulfilled_penalty = unfulfilled_orders * Config.ALPHA_UNFULFILLED

        return {
            'used_vehicles': used_vehicles,
            'vehicle_cost_raw': vehicle_cost_raw,
            'reject_penalty': reject_penalty,
            'unfulfilled_penalty': unfulfilled_penalty,
            'reject_count': reject_count,
            'active_rejected_orders': active_rejected_orders,
            'rejected_orders': active_rejected_orders,
            'completed_orders': completed_orders,
            'unserved_orders': unserved_orders,
            'unfulfilled_orders': unfulfilled_orders,
            'pickup_only_orders': pickup_only_orders,
            'delivery_without_pickup_orders': delivery_without_pickup_orders,
            'started_not_completed_orders': started_not_completed_orders,
            'pickup_only_mask': pickup_only_mask,
            'started_not_completed_mask': started_not_completed_mask,
            'trip_overtime_penalty': trip_overtime_penalty,
            'trip_overtime_hours_total': trip_overtime_total,
        }

    @staticmethod
    def get_costs(dataset, pi, return_details=False):
        """计算路径的训练目标与人民币口径成本分解（v5 重构后入口）。

        v5 主要修复：
        - cumulative_demand 在 depot 切换时强制清零（避免能耗计费跨车泄漏）
        - 真实落实 MAX_TRIP_TIME 软惩罚（trip_overtime_penalty）
        - 拆分为 _compute_distance_energy / _compute_time_and_delay / _compute_vehicle_and_penalty 三段，便于单测
        """
        batch_size = dataset['loc'].size(0)
        device = dataset['loc'].device

        graph_size = _extract_graph_size(dataset)
        normalization_profile = Config.get_normalization_profile(graph_size)
        reject_index = 2 * graph_size + 1
        reject_count = (pi == reject_index).sum(1).float() if pi.numel() > 0 else torch.zeros(batch_size, device=device)

        if (pi == reject_index).any():
            filtered_rows = []
            max_len = 1
            for row in pi:
                kept = row[row != reject_index]
                if kept.numel() == 0:
                    kept = row.new_zeros(1)
                filtered_rows.append(kept)
                max_len = max(max_len, kept.numel())
            pi_for_cost = pi.new_zeros(pi.size(0), max_len)
            for row_idx, kept in enumerate(filtered_rows):
                pi_for_cost[row_idx, :kept.numel()] = kept
        else:
            pi_for_cost = pi

        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), dim=1)

        depot_tw = torch.zeros(batch_size, 1, 2, device=device)
        depot_tw[:, :, 0] = Config.OPERATION_START
        depot_tw[:, :, 1] = Config.OPERATION_END
        tw_with_depot = torch.cat((depot_tw, dataset['time_windows']), dim=1)

        depot_type = torch.zeros(batch_size, 1, device=device)
        type_with_depot = torch.cat((depot_type, dataset['node_type']), dim=1)

        tw_in_order = tw_with_depot.gather(1, pi_for_cost[..., None].expand(-1, -1, 2))
        type_in_order = type_with_depot.gather(1, pi_for_cost)

        # 1) 距离与能耗
        energy_cost_raw, segment_distances, total_distance = MCVRPPDTW._compute_distance_energy(
            dataset, pi_for_cost, loc_with_depot
        )

        # 2) 时间仿真与延误（v6: 仅 delivery 节点累计延误）
        time_dict = MCVRPPDTW._compute_time_and_delay(pi_for_cost, segment_distances, tw_in_order, type_in_order, graph_size, dataset)
        time_dict['reject_count'] = reject_count

        # 3) 车辆数与软惩罚（v6: 移除 underload, 移除 time_penalty）
        vp_dict = MCVRPPDTW._compute_vehicle_and_penalty(pi_for_cost, graph_size, time_dict)

        passenger_delivery_delay_cost_raw = time_dict['passenger_delivery_delay_minutes'] * Config.PASSENGER_DELAY_COST
        cargo_delay_cost_raw = time_dict['cargo_delay_minutes'] * Config.CARGO_DELAY_COST

        # 归一化 + 加权（方案 B）
        normalized_energy_cost = energy_cost_raw / normalization_profile['energy_cost_raw']
        normalized_passenger_penalty = passenger_delivery_delay_cost_raw / normalization_profile['passenger_delivery_delay_cost_raw']
        normalized_cargo_delay = cargo_delay_cost_raw / normalization_profile['cargo_delay_cost_raw']
        vehicle_cost_norm = max(float(normalization_profile.get('vehicle_cost_norm', 1.0)), 1.0)
        normalized_vehicle_cost = vp_dict['vehicle_cost_raw'] / vehicle_cost_norm

        weighted_energy_cost = Config.ALPHA_ENERGY * normalized_energy_cost
        weighted_delay_cost = Config.ALPHA_DELAY * (normalized_passenger_penalty + normalized_cargo_delay)
        weighted_vehicle_cost = Config.ALPHA_VEHICLE * normalized_vehicle_cost

        # 训练目标 = 归一化加权 + 主动 reject 惩罚 + 未履约兜底 + 单趟超时软惩罚
        total_cost = (
            weighted_energy_cost
            + weighted_delay_cost
            + weighted_vehicle_cost
            + vp_dict['reject_penalty']
            + vp_dict['unfulfilled_penalty']
            + vp_dict['trip_overtime_penalty']
        )
        # v1.1: 报告目标 = 纯人民币口径
        total_cost_raw = (
            energy_cost_raw
            + passenger_delivery_delay_cost_raw
            + cargo_delay_cost_raw
            + vp_dict['vehicle_cost_raw']
            + vp_dict['reject_penalty']
            + vp_dict['unfulfilled_penalty']
            + vp_dict['trip_overtime_penalty']
        )

        if return_details:
            details = {
                'objective_total': total_cost,
                'total_cost': total_cost_raw,
                'total_cost_raw': total_cost_raw,
                'energy_cost_raw': energy_cost_raw,
                'passenger_delivery_delay_cost_raw': passenger_delivery_delay_cost_raw,
                'cargo_delay_cost_raw': cargo_delay_cost_raw,
                'vehicle_cost_raw': vp_dict['vehicle_cost_raw'],
                'reject_penalty': vp_dict['reject_penalty'],
                'unfulfilled_penalty': vp_dict['unfulfilled_penalty'],
                'trip_overtime_penalty': vp_dict['trip_overtime_penalty'],
                'trip_overtime_hours': vp_dict['trip_overtime_hours_total'],
                'active_rejected_orders': vp_dict['active_rejected_orders'],
                'rejected_orders': vp_dict['rejected_orders'],
                'completed_orders': vp_dict['completed_orders'],
                'unserved_orders': vp_dict['unserved_orders'],
                'unfulfilled_orders': vp_dict['unfulfilled_orders'],
                'pickup_only_orders': vp_dict['pickup_only_orders'],
                'delivery_without_pickup_orders': vp_dict['delivery_without_pickup_orders'],
                'started_not_completed_orders': vp_dict['started_not_completed_orders'],
                'normalized_energy_cost': normalized_energy_cost,
                'normalized_passenger_penalty': normalized_passenger_penalty,
                'normalized_cargo_delay_cost': normalized_cargo_delay,
                'normalized_vehicle_cost': normalized_vehicle_cost,
                'weighted_energy_cost': weighted_energy_cost,
                'weighted_delay_cost': weighted_delay_cost,
                'weighted_vehicle_cost': weighted_vehicle_cost,
                'used_vehicles': vp_dict['used_vehicles'],
                'total_distance': total_distance,
                'passenger_delivery_delay_minutes': time_dict['passenger_delivery_delay_minutes'],
                'cargo_delay_minutes': time_dict['cargo_delay_minutes'],
                'passenger_pickup_hard_violations': time_dict['passenger_pickup_violations'],
                'passenger_ride_time_violations': time_dict['passenger_ride_time_violations'],
                'passenger_total_ride_time_violations': time_dict['passenger_total_ride_time_violations'],
                'passenger_excess_ride_time_violations': time_dict['passenger_excess_ride_time_violations'],
                'passenger_ride_time_minutes_total': time_dict['passenger_ride_time_minutes_total'],
                'passenger_ride_time_minutes_max': time_dict['passenger_ride_time_minutes_max'],
                'passenger_excess_ride_time_minutes_total': time_dict['passenger_excess_ride_time_minutes_total'],
                'passenger_excess_ride_time_minutes_max': time_dict['passenger_excess_ride_time_minutes_max'],
                'passenger_pickup_total': time_dict['passenger_pickup_total'],
                'passenger_pickup_on_time': time_dict['passenger_pickup_on_time'],
                'passenger_delivery_total': time_dict['passenger_delivery_total'],
                'passenger_delivery_on_time': time_dict['passenger_delivery_on_time'],
                'cargo_delivery_total': time_dict['cargo_delivery_total'],
                'cargo_delivery_on_time': time_dict['cargo_delivery_on_time'],
                'trip_time_hours': time_dict['final_time'] - Config.OPERATION_START,
                # 兼容旧字段
                'energy_cost': energy_cost_raw,
                'passenger_delay_cost': passenger_delivery_delay_cost_raw,
                'passenger_delay_minutes': time_dict['passenger_delivery_delay_minutes'],
                'passenger_penalty_raw': passenger_delivery_delay_cost_raw,
                'cargo_delay_cost': cargo_delay_cost_raw,
                'vehicle_cost': vp_dict['vehicle_cost_raw'],
            }
            return total_cost, details

        return total_cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MCVRPPDTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        from state_mcvrptw_v2 import StateMCVRPPDTW
        return StateMCVRPPDTW.initialize(*args, **kwargs)


def _extract_graph_size(dataset):
    """从 dataset 中解析图规模 N（订单数），兼容 tensor/scalar。"""
    # 优先使用 n_orders 字段，如果不存在则退化为 loc.shape[1]//2
    graph_size = dataset.get('n_orders', None)
    if graph_size is None:
        # loc: [B, 2N, 2] -> N = loc.size(1)//2
        loc = dataset['loc']
        if torch.is_tensor(loc):
            return int(loc.size(1) // 2)
        return int(len(loc) // 2)

    if torch.is_tensor(graph_size):
        # 既兼容标量 tensor，也兼容 batch tensor，这里统一取第一个样本
        graph_size = graph_size.reshape(-1)[0].item()
    return int(graph_size)


def node_routes_to_pi(routes, n_orders, max_seq_len=None):
    """将多车辆节点路径列表转换为 get_costs 可识别的单序列 pi。

    约定：
    - routes 是一个长度为 K 的列表，每个元素是该车的节点序列（不含 depot 0）
    - 编码规则：
        全序列以一个 0 作为起点；
        依次拼接每辆车的节点序列；
        每辆车的行程结束都用 0 结束；
        因为不再存在 CHARGING_DEPOT，多趟行程仅通过 0 作为分隔符和终止符。

    - max_seq_len (可选)：
        如果指定，则对 pi 进行截断或 padding；padding 使用 0，确保最后一个元素为 0。
    """
    depot = 0

    if not routes:
        # 没有任何车辆路径，至少返回 [0, 0]
        pi = torch.tensor([depot, depot], dtype=torch.long)
    else:
        # 清洗：去掉 route 中多余的 0，并滤空
        cleaned_routes = []
        for route in routes:
            customer_nodes = [int(node) for node in route if int(node) != depot]
            if customer_nodes:
                cleaned_routes.append(customer_nodes)

        if not cleaned_routes:
            pi = torch.tensor([depot, depot], dtype=torch.long)
        else:
            flattened = [depot]
            for route_idx, route in enumerate(cleaned_routes):
                flattened.extend(route)
                # 每辆车行程结尾都加一个 0，表示该车下班/行程结束
                flattened.append(depot)
            pi = torch.tensor(flattened, dtype=torch.long)

    if max_seq_len is not None:
        if pi.numel() < max_seq_len:
            pad = torch.full((max_seq_len - pi.numel(),), depot, dtype=torch.long)
            pi = torch.cat((pi, pad), dim=0)
        else:
            pi = pi[:max_seq_len]
            if pi[-1].item() != depot:
                pi[-1] = depot

    return pi.unsqueeze(0)


def calibrate_normalization(
    graph_size,
    num_samples=256,
    seed=1234,
    num_vehicles=None,
    output_path=None,
):
    """基于随机订单顺序校准不同图规模的归一化常数。"""
    graph_size = int(graph_size)
    num_vehicles = num_vehicles or Config.get_default_num_vehicles(graph_size)
    dataset = MCVRPPDTWDataset(num_samples=num_samples, graph_size=graph_size, seed=seed)
    rng = np.random.default_rng(seed)

    energy_values = []
    passenger_values = []
    cargo_values = []
    vehicle_values = []

    for sample in dataset:
        order_sequence = rng.permutation(graph_size).tolist()
        partitions = np.array_split(order_sequence, num_vehicles)
        routes = []
        for partition in partitions:
            route = []
            for order_idx in partition.tolist():
                pickup_node = order_idx + 1
                delivery_node = order_idx + graph_size + 1
                route.extend([pickup_node, delivery_node])
            if route:
                routes.append(route)

        batch = {
            key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in sample.items()
        }
        pi = node_routes_to_pi(routes, graph_size)
        _, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)
        energy_values.append(details['energy_cost_raw'].item())
        passenger_values.append(details['passenger_delivery_delay_cost_raw'].item())
        cargo_values.append(details['cargo_delay_cost_raw'].item())
        vehicle_values.append(details['vehicle_cost_raw'].item())

    profile = Config.set_normalization_profile(graph_size, {
        'graph_size': graph_size,
        'energy_cost_raw': float(np.mean(energy_values)) if energy_values else 1.0,
        'passenger_delivery_delay_cost_raw': float(np.mean(passenger_values)) if passenger_values else 1.0,
        'cargo_delay_cost_raw': float(np.mean(cargo_values)) if cargo_values else 1.0,
        'vehicle_cost_norm': float(np.mean(vehicle_values)) if vehicle_values else 1.0,
        'num_samples': int(num_samples),
        'seed': int(seed),
        'num_vehicles': int(num_vehicles),
    })

    output_path = output_path or Config.get_normalization_output_path(graph_size)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as profile_file:
        json.dump(profile, profile_file, indent=2, ensure_ascii=False)

    return profile


class MCVRPPDTWDataset(Dataset):
    """
    客货混合Pickup-Delivery VRP数据集

    数据生成策略（v1.1）：
    1. 50%节点随机均匀分布，50%节点聚类分布
    2. 乘客订单按 PASSENGER_RATIO=0.6 概率独立采样
    3. 每个订单包含一个pickup点和一个delivery点
    4. 乘客时间窗 1.0 小时，货物时间窗 1.0 小时
    5. 需求量：乘客 80% 为 {1,2} / 20% 为 {3,4}；货物 1-3 单位
    """
    
    def __init__(
        self,
        num_samples=10000,
        graph_size=50,  # 订单数量（实际节点数=2*graph_size）
        seed=1234,
        distribution='mixed',  # 'random', 'cluster', 'mixed'
        passenger_ratio_override=None,
        passenger_distance_mix_override=None,
        passenger_tw_period_weights_override=None,
        cargo_tw_period_weights_override=None,
    ):
        """初始化。

        Args:
            num_samples: 数据集样本数量
            graph_size: 订单数量（pickup-delivery对数）
            seed: 随机种子
            distribution: 分布类型
        """
        super().__init__()
        
        self.num_samples = num_samples
        self.graph_size = graph_size  # 订单数
        self.n_nodes = 2 * graph_size  # 实际节点数
        self.distribution = distribution
        self.passenger_ratio_override = passenger_ratio_override
        self.passenger_distance_mix_override = passenger_distance_mix_override
        self.passenger_tw_period_weights_override = passenger_tw_period_weights_override
        self.cargo_tw_period_weights_override = cargo_tw_period_weights_override

        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.data = self._generate_data()
    
    def _generate_locations(self, n_points):
        """生成节点位置（50%随机 + 50%聚类）。

        Returns:
            locations: [n_points, 2]，归一化坐标 [0,1]
        """
        n_random = int(n_points * Config.RANDOM_RATIO)
        n_cluster = n_points - n_random
        
        # 随机分布节点
        random_locs = torch.rand(n_random, 2)
        
        if n_cluster > 0:
            # 生成聚类中心
            centers = torch.rand(Config.NUM_CLUSTERS, 2)
            
            # 为每个聚类节点分配中心
            cluster_assignments = torch.randint(0, Config.NUM_CLUSTERS, (n_cluster,))
            
            # 在中心周围生成节点（高斯分布）
            cluster_locs = centers[cluster_assignments] + torch.randn(n_cluster, 2) * 0.1
            cluster_locs = torch.clamp(cluster_locs, 0.0, 1.0)
            
            # 合并
            locations = torch.cat([random_locs, cluster_locs], dim=0)
            # 随机打乱顺序
            perm = torch.randperm(n_points)
            locations = locations[perm]
        else:
            locations = random_locs
        
        return locations
    
    def _generate_data(self):
        """生成所有样本数据。"""
        data_list = []
        
        for _ in range(self.num_samples):
            # 生成depot（车场，位于区域中心附近）
            depot = torch.tensor([0.5, 0.5]) + torch.randn(2) * 0.1
            depot = torch.clamp(depot, 0.1, 0.9)
            
            # 生成pickup和delivery点位置
            n_orders = self.graph_size
            pickup_locs = self._generate_locations(n_orders)

            # 分配订单类型：v1.1 固定乘客 60% / 货物 40%，curriculum 可覆盖 passenger ratio
            default_passenger_ratio = Config.PASSENGER_RATIO_LARGE if n_orders >= 50 else Config.PASSENGER_RATIO
            passenger_ratio = float(self.passenger_ratio_override) if self.passenger_ratio_override is not None else default_passenger_ratio
            passenger_ratio = min(max(passenger_ratio, 0.0), 1.0)
            order_type = (torch.rand(n_orders) < passenger_ratio).float()

            cargo_short_distance_count = int(round(n_orders * Config.PD_SHORT_DISTANCE_RATIO))
            cargo_short_mask = torch.zeros(n_orders, dtype=torch.bool)
            if cargo_short_distance_count > 0:
                cargo_short_mask[torch.randperm(n_orders)[:cargo_short_distance_count]] = True
            cargo_max_distance = torch.where(
                cargo_short_mask,
                torch.full((n_orders,), Config.PD_SHORT_DISTANCE_KM / Config.AREA_SIZE),
                torch.full((n_orders,), Config.PD_LONG_DISTANCE_KM / Config.AREA_SIZE),
            )

            passenger_indices = torch.nonzero(order_type == 1, as_tuple=False).squeeze(-1)
            passenger_count = passenger_indices.numel()
            passenger_max_distance = torch.zeros(n_orders)
            passenger_min_distance = torch.zeros(n_orders)
            if passenger_count > 0:
                shuffled_passenger = passenger_indices[torch.randperm(passenger_count)]
                if self.passenger_distance_mix_override is None:
                    short_ratio = Config.PASSENGER_PD_SHORT_RATIO
                    mid_ratio = Config.PASSENGER_PD_MID_RATIO
                    long_ratio = Config.PASSENGER_PD_LONG_RATIO
                else:
                    short_ratio, mid_ratio, long_ratio = self.passenger_distance_mix_override
                long_count = int(round(passenger_count * long_ratio))
                mid_count = int(round(passenger_count * mid_ratio))
                short_count = max(passenger_count - long_count - mid_count, 0)
                long_indices = shuffled_passenger[:long_count]
                mid_indices = shuffled_passenger[long_count:long_count + mid_count]
                short_indices = shuffled_passenger[long_count + mid_count:long_count + mid_count + short_count]

                if long_indices.numel() > 0:
                    passenger_min_distance[long_indices] = Config.PASSENGER_PD_LONG_MIN_KM / Config.AREA_SIZE
                    passenger_max_distance[long_indices] = Config.PASSENGER_PD_LONG_MAX_KM / Config.AREA_SIZE
                if mid_indices.numel() > 0:
                    passenger_min_distance[mid_indices] = Config.PASSENGER_PD_MID_MIN_KM / Config.AREA_SIZE
                    passenger_max_distance[mid_indices] = Config.PASSENGER_PD_MID_MAX_KM / Config.AREA_SIZE
                if short_indices.numel() > 0:
                    passenger_min_distance[short_indices] = Config.PASSENGER_PD_SHORT_MIN_KM / Config.AREA_SIZE
                    passenger_max_distance[short_indices] = Config.PASSENGER_PD_SHORT_MAX_KM / Config.AREA_SIZE

            delivery_locs = pickup_locs.clone()
            for order_idx in range(n_orders):
                if order_type[order_idx] == 1:
                    min_radius = passenger_min_distance[order_idx].item()
                    max_radius = passenger_max_distance[order_idx].item()
                else:
                    min_radius = 0.0
                    max_radius = cargo_max_distance[order_idx].item()

                for _ in range(80):
                    candidate = pickup_locs[order_idx] + torch.randn(2) * max(max_radius / 2.5, 1e-3)
                    candidate = torch.clamp(candidate, 0.0, 1.0)
                    candidate_distance = torch.norm(candidate - pickup_locs[order_idx]).item()
                    if (candidate_distance <= max_radius + 1e-6) and (candidate_distance >= max(min_radius - 1e-6, 0.0)):
                        delivery_locs[order_idx] = candidate
                        break
                else:
                    angle = torch.rand(1).item() * 2 * math.pi
                    radius = max((min_radius + max_radius) * 0.5, max_radius * 0.8)
                    candidate = pickup_locs[order_idx] + torch.tensor([math.cos(angle), math.sin(angle)]) * radius
                    delivery_locs[order_idx] = torch.clamp(candidate, 0.0, 1.0)

            # 合并所有节点：[pickup_1, ..., pickup_n, delivery_1, ..., delivery_n]
            loc = torch.cat([pickup_locs, delivery_locs], dim=0)
            
            # 节点类型（pickup和delivery继承订单类型）
            node_type = torch.cat([order_type, order_type], dim=0)
            
            # 生成需求量：乘客订单 80% 为 1-2 人，其余为 3-4 人；货物为 1-3 单位
            demands = torch.randint(Config.DEMAND_MIN, Config.DEMAND_MAX + 1, (n_orders,)).float()
            passenger_mask = order_type == 1
            passenger_indices = torch.nonzero(passenger_mask, as_tuple=False).squeeze(-1)
            if passenger_indices.numel() > 0:
                small_group_count = int(round(passenger_indices.numel() * Config.PASSENGER_SMALL_GROUP_RATIO))
                shuffled = passenger_indices[torch.randperm(passenger_indices.numel())]
                small_indices = shuffled[:small_group_count]
                large_indices = shuffled[small_group_count:]
                if small_indices.numel() > 0:
                    demands[small_indices] = torch.randint(1, 3, (small_indices.numel(),), dtype=torch.int64).float()
                if large_indices.numel() > 0:
                    demands[large_indices] = torch.randint(3, 5, (large_indices.numel(),), dtype=torch.int64).float()
            
            # 归一化需求（相对于容量）
            demand_passenger = torch.zeros(self.n_nodes)
            demand_cargo = torch.zeros(self.n_nodes)
            
            for i in range(n_orders):
                if order_type[i] == 1:  # 乘客
                    demand_passenger[i] = demands[i] / Config.PASSENGER_CAPACITY
                    demand_passenger[i + n_orders] = -demands[i] / Config.PASSENGER_CAPACITY  # delivery为负（释放容量）
                else:  # 货物
                    demand_cargo[i] = demands[i] / Config.CARGO_CAPACITY
                    demand_cargo[i + n_orders] = -demands[i] / Config.CARGO_CAPACITY
            
            # 生成时间窗
            time_windows = self._generate_time_windows(depot, loc, node_type, n_orders)
            
            # Pickup-Delivery配对信息
            # pickup_idx[i] = i, delivery_idx[i] = i + n_orders
            pickup_delivery_pairs = torch.arange(n_orders)
            
            pd_pair_mask = torch.zeros(self.n_nodes + 1, self.n_nodes + 1, dtype=torch.bool)
            for order_idx in range(n_orders):
                pickup_node = order_idx + 1
                delivery_node = order_idx + n_orders + 1
                pd_pair_mask[pickup_node, delivery_node] = True
                pd_pair_mask[delivery_node, pickup_node] = True

            data_list.append({
                'depot': depot,
                'loc': loc,
                'node_type': node_type,
                'demand_passenger': demand_passenger,
                'demand_cargo': demand_cargo,
                'time_windows': time_windows,
                'pickup_delivery_pairs': pickup_delivery_pairs,
                'pd_pair_mask': pd_pair_mask,
                'n_orders': n_orders,
            })
        
        return data_list
    
    def _generate_time_windows(self, depot, loc, node_type, n_orders):
        """生成时间窗（分时段混合采样，保证下午订单占比下限）。"""
        n_nodes = loc.size(0)
        time_windows = torch.zeros(n_nodes, 2)

        dist_from_depot = (loc - depot).norm(p=2, dim=1) * Config.AREA_SIZE  # km
        travel_time_from_depot = dist_from_depot / Config.VEHICLE_SPEED  # 小时

        period_bounds = (
            (10.0, 12.0),
            (12.0, 14.0),
            (14.0, 16.0),
        )
        passenger_tw_weights = self.passenger_tw_period_weights_override or Config.PASSENGER_TW_PERIOD_WEIGHTS
        cargo_tw_weights = self.cargo_tw_period_weights_override or Config.CARGO_TW_PERIOD_WEIGHTS
        passenger_probs = torch.tensor(passenger_tw_weights, dtype=torch.float)
        cargo_probs = torch.tensor(cargo_tw_weights, dtype=torch.float)
        passenger_probs = passenger_probs / passenger_probs.sum()
        cargo_probs = cargo_probs / cargo_probs.sum()

        for i in range(n_orders):
            earliest_arrival = Config.OPERATION_START + travel_time_from_depot[i]
            tw_width = Config.PASSENGER_TW_WIDTH if node_type[i] == 1 else Config.CARGO_TW_WIDTH

            feasible_start_low = max(Config.OPERATION_START, float(earliest_arrival))
            feasible_start_high = Config.OPERATION_END - tw_width

            probs = passenger_probs if node_type[i] == 1 else cargo_probs
            period_idx = int(torch.multinomial(probs, 1).item())

            period_low_raw, period_high_raw = period_bounds[period_idx]
            period_low = max(feasible_start_low, period_low_raw)
            period_high = min(feasible_start_high, period_high_raw)

            if period_high > period_low:
                tw_start = period_low + torch.rand(1).item() * (period_high - period_low)
            elif feasible_start_high > feasible_start_low:
                tw_start = feasible_start_low + torch.rand(1).item() * (feasible_start_high - feasible_start_low)
            else:
                tw_start = feasible_start_low

            tw_start = min(max(tw_start, Config.OPERATION_START), Config.OPERATION_END - tw_width)
            tw_end = tw_start + tw_width

            time_windows[i, 0] = tw_start
            time_windows[i, 1] = tw_end

            pickup_to_delivery_time = (loc[i] - loc[i + n_orders]).norm(p=2) * Config.AREA_SIZE / Config.VEHICLE_SPEED
            delivery_earliest = tw_start + Config.SERVICE_TIME + pickup_to_delivery_time
            delivery_latest = tw_end + Config.SERVICE_TIME + pickup_to_delivery_time + tw_width
            delivery_earliest = min(delivery_earliest, Config.OPERATION_END)
            delivery_latest = min(delivery_latest, Config.OPERATION_END)
            if delivery_latest < delivery_earliest:
                delivery_latest = delivery_earliest

            time_windows[i + n_orders, 0] = delivery_earliest
            time_windows[i + n_orders, 1] = delivery_latest

        return time_windows
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def test_dataset():
    """测试数据集与成本计算的简单 dry run。"""
    print("=" * 60)
    print("客货混合Pickup-Delivery VRP数据集测试")
    print("=" * 60)
    
    # 生成小规模测试数据
    dataset = MCVRPPDTWDataset(num_samples=2, graph_size=10)
    print(f"\n数据集大小: {len(dataset)}")
    print(f"订单数量: {dataset.graph_size}")
    print(f"节点数量: {dataset.n_nodes}")
    
    sample = dataset[0]
    print(f"\n第一个样本:")
    print(f"  Depot坐标: {sample['depot']}")
    print(f"  节点坐标形状: {sample['loc'].shape}")
    print(f"  节点类型（前10个）: {sample['node_type'][:10]}")
    
    # 构造一个简单的多车多行程节点路径，并转换为 pi
    n_orders = int(sample['n_orders'])
    node_routes = []
    # 第一辆车跑前半部分订单
    route1 = []
    for order_id in range(n_orders // 2):
        route1.extend([order_id + 1, order_id + 1 + n_orders])
    node_routes.append(route1)
    # 第二辆车跑剩余订单
    route2 = []
    for order_id in range(n_orders // 2, n_orders):
        route2.extend([order_id + 1, order_id + 1 + n_orders])
    node_routes.append(route2)

    pi = node_routes_to_pi(node_routes, n_orders)
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in sample.items()}
    costs, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)
    print("\n简单 dry run 成本:")
    print(f"  训练目标: {costs.mean().item():.4f}")
    print(f"  原始总成本: {details['total_cost_raw'].mean().item():.4f} 元")
    print(f"  能耗成本: {details['energy_cost_raw'].mean().item():.4f} 元")
    print(f"  乘客 delivery 延误成本: {details['passenger_delivery_delay_cost_raw'].mean().item():.4f} 元")
    print(f"  货物延误成本: {details['cargo_delay_cost_raw'].mean().item():.4f} 元")
    print(f"  单趟超时惩罚: {details['trip_overtime_penalty'].mean().item():.4f} 元")
    print(f"  主动拒单惩罚: {details['reject_penalty'].mean().item():.4f} 元")
    print(f"  未履约兜底惩罚: {details['unfulfilled_penalty'].mean().item():.4f} 元")
    print(f"  车辆成本: {details['vehicle_cost_raw'].mean().item():.4f} 元, 使用车辆数: {details['used_vehicles'].mean().item():.2f}")
    print(f"  总行驶距离: {details['total_distance'].mean().item():.4f} km")
    print(f"  行程时间: {details['trip_time_hours'].mean().item():.4f} 小时")
    
    print("\n测试完成！")


if __name__ == '__main__':
    test_dataset()
