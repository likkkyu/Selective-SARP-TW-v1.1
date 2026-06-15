"""
Pickup-Delivery约束的状态类
State for Multi-Compartment VRP with Pickup-Delivery and Time Windows

关键约束：
1. Pickup-Delivery先序约束：必须先访问pickup点，才能访问对应的delivery点
2. 双舱室容量约束：乘客和货物分别计算容量
3. 3小时单程时间约束
4. passenger pickup 硬时间窗 + passenger ride-time 硬约束
5. 新增可学习 reject 动作：动作空间扩展为 N+2，state 负责确定性标记被拒订单
"""

import torch
from typing import NamedTuple

from problem_mcvrptw_v2 import Config


class StateMCVRPPDTW(NamedTuple):
    """Pickup-Delivery VRPTW状态表示。"""

    coords: torch.Tensor
    node_type: torch.Tensor
    demand_passenger: torch.Tensor
    demand_cargo: torch.Tensor
    time_windows: torch.Tensor
    n_orders: int
    ids: torch.Tensor
    current_time: torch.Tensor
    trip_start_time: torch.Tensor
    prev_a: torch.Tensor
    used_capacity_passenger: torch.Tensor
    used_capacity_cargo: torch.Tensor
    used_vehicles: torch.Tensor
    visited_: torch.Tensor
    picked_up_: torch.Tensor
    passenger_pickup_time: torch.Tensor
    rejected_: torch.Tensor
    reject_count: torch.Tensor
    allow_reject: bool
    max_concurrent_open_orders: int
    enable_delivery_viability: bool
    enable_viability_fallback: bool
    relax_pickup_commitment_trip_time: bool
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    deadlock_count: torch.Tensor
    deadlock_limit: torch.Tensor
    terminal_: torch.Tensor
    i: torch.Tensor

    PASSENGER_CAPACITY = 1.0
    CARGO_CAPACITY = 1.0
    VEHICLE_SPEED = Config.VEHICLE_SPEED
    AREA_SIZE = Config.AREA_SIZE
    MAX_TRIP_TIME = Config.MAX_TRIP_TIME
    OPERATION_START = Config.OPERATION_START
    OPERATION_END = Config.OPERATION_END
    SERVICE_TIME = Config.SERVICE_TIME

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        return self.visited_.to(torch.uint8)

    @property
    def reject_index(self):
        return 2 * self.n_orders + 1

    def get_open_started_mask(self):
        if self.n_orders == 0:
            return torch.zeros(self.ids.size(0), 0, dtype=torch.bool, device=self.coords.device)
        delivered = self.visited[:, :, self.n_orders + 1:2 * self.n_orders + 1].bool().squeeze(1)
        return self.picked_up_.bool().squeeze(1) & (~delivered) & (~self.rejected_.bool().squeeze(1))

    def get_open_started_count(self):
        return self.get_open_started_mask().sum(-1, keepdim=True)

    def has_open_started_orders(self):
        return self.get_open_started_count() > 0

    def _delivery_sequence_feasible(self, start_coord, start_time, trip_start_time, delivery_sequence,
                                    delivery_coords, delivery_earliest, delivery_to_depot_time,
                                    passenger_orders, passenger_pickup_times, direct_ride_time,
                                    return_reason=False, ignore_trip_time=False):
        current_coord = start_coord
        current_time = float(start_time)
        trip_start = float(trip_start_time)
        for order_idx in delivery_sequence:
            order_idx = int(order_idx)
            delivery_coord = delivery_coords[order_idx]
            travel_time = float(((delivery_coord - current_coord).norm(p=2) * self.AREA_SIZE / self.VEHICLE_SPEED).item())
            arrival_time = current_time + travel_time
            if passenger_orders[order_idx].item():
                pickup_time = float(passenger_pickup_times[order_idx].item())
                if pickup_time < 0:
                    return (False, 'missing_pickup_time') if return_reason else False
                ride_time = arrival_time - pickup_time
                excess_ride_time = max(ride_time - float(direct_ride_time[order_idx].item()), 0.0)
                if Config.HARD_PASSENGER_MAX_RIDE_TIME and (
                    ride_time > Config.PASSENGER_MAX_RIDE_TIME_MINUTES / 60.0 + 1e-5
                    or excess_ride_time > Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES / 60.0 + 1e-5
                ):
                    return (False, 'ride_time') if return_reason else False
            service_start = max(arrival_time, float(delivery_earliest[order_idx].item()))
            finish_time = service_start + self.SERVICE_TIME
            finish_with_return = finish_time + float(delivery_to_depot_time[order_idx].item())
            if (not ignore_trip_time) and Config.HARD_MAX_TRIP_TIME and (finish_with_return - trip_start > self.MAX_TRIP_TIME + 1e-5):
                return (False, 'trip_time') if return_reason else False
            if Config.HARD_OPERATION_END and (finish_with_return > self.OPERATION_END + 1e-5):
                return (False, 'ops_end') if return_reason else False
            current_coord = delivery_coord
            current_time = finish_time
        return (True, None) if return_reason else True

    def _has_feasible_open_completion(self, start_coord, start_time, trip_start_time, open_mask,
                                      delivery_coords, delivery_earliest, delivery_to_depot_time,
                                      passenger_orders, passenger_pickup_times, direct_ride_time,
                                      return_reason=False, ignore_trip_time=False):
        open_indices = torch.nonzero(open_mask, as_tuple=False).squeeze(-1)
        if open_indices.numel() == 0:
            return (True, None) if return_reason else True
        if open_indices.numel() > 6:
            return (False, 'open_over_6') if return_reason else False
        order_list = open_indices.tolist()
        if open_indices.numel() == 1:
            return self._delivery_sequence_feasible(
                start_coord, start_time, trip_start_time, order_list,
                delivery_coords, delivery_earliest, delivery_to_depot_time,
                passenger_orders, passenger_pickup_times, direct_ride_time,
                return_reason=return_reason,
                ignore_trip_time=ignore_trip_time,
            )
        from itertools import permutations
        failure_reasons = set()
        for sequence in permutations(order_list):
            result = self._delivery_sequence_feasible(
                start_coord, start_time, trip_start_time, sequence,
                delivery_coords, delivery_earliest, delivery_to_depot_time,
                passenger_orders, passenger_pickup_times, direct_ride_time,
                return_reason=return_reason,
                ignore_trip_time=ignore_trip_time,
            )
            if return_reason:
                feasible, reason = result
                if feasible:
                    return True, None
                if reason is not None:
                    failure_reasons.add(reason)
            elif result:
                return True
        if not return_reason:
            return False
        if not failure_reasons:
            return False, 'unknown'
        if len(failure_reasons) == 1:
            return False, next(iter(failure_reasons))
        return False, 'mixed'

    def _delivery_step_feasible(self, start_coord, start_time, trip_start_time, order_idx,
                                delivery_coords, delivery_earliest, delivery_to_depot_time,
                                passenger_orders, passenger_pickup_times, direct_ride_time):
        delivery_coord = delivery_coords[order_idx]
        current_time = float(start_time)
        trip_start = float(trip_start_time)
        travel_time = float(((delivery_coord - start_coord).norm(p=2) * self.AREA_SIZE / self.VEHICLE_SPEED).item())
        arrival_time = current_time + travel_time
        if passenger_orders[order_idx].item():
            pickup_time = float(passenger_pickup_times[order_idx].item())
            if pickup_time < 0:
                return False, None, None
            ride_time = arrival_time - pickup_time
            excess_ride_time = max(ride_time - float(direct_ride_time[order_idx].item()), 0.0)
            if Config.HARD_PASSENGER_MAX_RIDE_TIME and (
                ride_time > Config.PASSENGER_MAX_RIDE_TIME_MINUTES / 60.0 + 1e-5
                or excess_ride_time > Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES / 60.0 + 1e-5
            ):
                return False, None, None
        service_start = max(arrival_time, float(delivery_earliest[order_idx].item()))
        finish_time = service_start + self.SERVICE_TIME
        finish_with_return = finish_time + float(delivery_to_depot_time[order_idx].item())
        if Config.HARD_MAX_TRIP_TIME and (finish_with_return - trip_start > self.MAX_TRIP_TIME + 1e-5):
            return False, None, None
        if Config.HARD_OPERATION_END and (finish_with_return > self.OPERATION_END + 1e-5):
            return False, None, None
        return True, delivery_coord, finish_time

    def _has_any_physical_delivery_step(self, start_coord, start_time, trip_start_time, open_mask,
                                        delivery_coords, delivery_earliest, delivery_to_depot_time,
                                        passenger_orders, passenger_pickup_times, direct_ride_time):
        open_indices = torch.nonzero(open_mask, as_tuple=False).squeeze(-1)
        if open_indices.numel() == 0:
            return True
        for order_idx_tensor in open_indices:
            order_idx = int(order_idx_tensor.item())
            feasible_step, _, _ = self._delivery_step_feasible(
                start_coord,
                start_time,
                trip_start_time,
                order_idx,
                delivery_coords,
                delivery_earliest,
                delivery_to_depot_time,
                passenger_orders,
                passenger_pickup_times,
                direct_ride_time,
            )
            if feasible_step:
                return True
        return False

    def _get_legal_delivery_orders(self, start_coord, start_time, trip_start_time, open_mask,
                                   delivery_coords, delivery_earliest, delivery_to_depot_time,
                                   passenger_orders, passenger_pickup_times, direct_ride_time,
                                   allow_fallback=False):
        open_indices = torch.nonzero(open_mask, as_tuple=False).squeeze(-1)
        if open_indices.numel() == 0:
            return [], [], False

        physical_orders = []
        viable_orders = []
        for order_idx_tensor in open_indices:
            order_idx = int(order_idx_tensor.item())
            feasible_step, next_coord, finish_time = self._delivery_step_feasible(
                start_coord,
                start_time,
                trip_start_time,
                order_idx,
                delivery_coords,
                delivery_earliest,
                delivery_to_depot_time,
                passenger_orders,
                passenger_pickup_times,
                direct_ride_time,
            )
            if not feasible_step:
                continue
            physical_orders.append(order_idx)
            open_after = open_mask.clone()
            open_after[order_idx] = False
            if self._has_legal_delivery_path(
                next_coord,
                finish_time,
                trip_start_time,
                open_after,
                delivery_coords,
                delivery_earliest,
                delivery_to_depot_time,
                passenger_orders,
                passenger_pickup_times,
                direct_ride_time,
            ):
                viable_orders.append(order_idx)

        used_fallback = bool(allow_fallback and len(viable_orders) == 0 and len(physical_orders) > 0)
        legal_orders = physical_orders if used_fallback else viable_orders
        return legal_orders, physical_orders, used_fallback

    def _has_legal_delivery_path(self, start_coord, start_time, trip_start_time, open_mask,
                                 delivery_coords, delivery_earliest, delivery_to_depot_time,
                                 passenger_orders, passenger_pickup_times, direct_ride_time):
        open_indices = torch.nonzero(open_mask, as_tuple=False).squeeze(-1)
        if open_indices.numel() == 0:
            return True
        legal_orders, _, _ = self._get_legal_delivery_orders(
            start_coord,
            start_time,
            trip_start_time,
            open_mask,
            delivery_coords,
            delivery_earliest,
            delivery_to_depot_time,
            passenger_orders,
            passenger_pickup_times,
            direct_ride_time,
            allow_fallback=False,
        )
        return len(legal_orders) > 0

    @staticmethod
    def initialize(
        input_data,
        visited_dtype=torch.uint8,
        allow_reject=True,
        deadlock_limit=2,
        max_concurrent_open_orders=1,
        enable_delivery_viability=False,
        enable_viability_fallback=False,
        relax_pickup_commitment_trip_time=False,
    ):
        depot = input_data['depot']
        loc = input_data['loc']
        node_type = input_data['node_type']
        demand_passenger = input_data['demand_passenger']
        demand_cargo = input_data['demand_cargo']
        time_windows = input_data['time_windows']

        batch_size, n_nodes, _ = loc.size()
        n_orders = n_nodes // 2
        device = loc.device

        coords = torch.cat((depot[:, None, :], loc), dim=1)

        depot_tw = torch.zeros(batch_size, 1, 2, device=device)
        depot_tw[:, :, 0] = StateMCVRPPDTW.OPERATION_START
        depot_tw[:, :, 1] = StateMCVRPPDTW.OPERATION_END
        tw_with_depot = torch.cat((depot_tw, time_windows), dim=1)

        depot_type = torch.zeros(batch_size, 1, device=device)
        type_with_depot = torch.cat((depot_type, node_type), dim=1)

        depot_demand = torch.zeros(batch_size, 1, device=device)
        demand_p_with_depot = torch.cat((depot_demand, demand_passenger), dim=1)
        demand_c_with_depot = torch.cat((depot_demand, demand_cargo), dim=1)

        visited_shape = (batch_size, 1, 2 * n_orders + 2)

        return StateMCVRPPDTW(
            coords=coords,
            node_type=type_with_depot,
            demand_passenger=demand_p_with_depot,
            demand_cargo=demand_c_with_depot,
            time_windows=tw_with_depot,
            n_orders=n_orders,
            ids=torch.arange(batch_size, dtype=torch.int64, device=device)[:, None],
            current_time=torch.full((batch_size, 1), StateMCVRPPDTW.OPERATION_START, device=device),
            trip_start_time=torch.full((batch_size, 1), StateMCVRPPDTW.OPERATION_START, device=device),
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            used_capacity_passenger=torch.zeros(batch_size, 1, device=device),
            used_capacity_cargo=torch.zeros(batch_size, 1, device=device),
            used_vehicles=torch.zeros(batch_size, 1, device=device),
            visited_=torch.zeros(visited_shape, dtype=torch.uint8, device=device),
            picked_up_=torch.zeros(batch_size, 1, n_orders, dtype=torch.uint8, device=device),
            passenger_pickup_time=torch.full((batch_size, 1, n_orders), -1.0, device=device),
            rejected_=torch.zeros(batch_size, 1, n_orders, dtype=torch.uint8, device=device),
            reject_count=torch.zeros(batch_size, 1, device=device),
            allow_reject=bool(allow_reject),
            max_concurrent_open_orders=max(int(max_concurrent_open_orders), 1),
            enable_delivery_viability=bool(enable_delivery_viability),
            enable_viability_fallback=bool(enable_viability_fallback),
            relax_pickup_commitment_trip_time=bool(relax_pickup_commitment_trip_time),
            lengths=torch.zeros(batch_size, 1, device=device),
            cur_coord=depot[:, None, :],
            deadlock_count=torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            deadlock_limit=torch.full((batch_size, 1), max(int(deadlock_limit), 1), dtype=torch.int64, device=device),
            terminal_=torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            i=torch.zeros(batch_size, 1, dtype=torch.int64, device=device),
        )

    def _active_views(self):
        ids_flat = self.ids.squeeze(-1)
        return (
            ids_flat,
            self.coords[ids_flat],
            self.node_type[ids_flat],
            self.time_windows[ids_flat],
            self.demand_passenger[ids_flat],
            self.demand_cargo[ids_flat],
        )

    def _deterministic_reject_order(self, mask=None):
        batch_size = self.ids.size(0)
        device = self.coords.device
        n_orders = self.n_orders
        if n_orders == 0:
            return torch.full((batch_size,), -1, dtype=torch.long, device=device)

        pickup_visited = self.visited[:, :, 1:n_orders + 1].bool().squeeze(1)
        delivery_visited = self.visited[:, :, n_orders + 1:2 * n_orders + 1].bool().squeeze(1)
        candidate_pickups = (~pickup_visited) & (~delivery_visited)
        candidate_pickups = candidate_pickups & (self.rejected_.squeeze(1) == 0)
        candidate_pickups = candidate_pickups & (self.picked_up_.squeeze(1) == 0)

        ids_flat = self.ids.squeeze(-1)
        tw_late = self.time_windows[ids_flat, 1:n_orders + 1, 1]
        inf = torch.full_like(tw_late, float('inf'))
        tw_score = torch.where(candidate_pickups, tw_late, inf)
        order_idx = tw_score.argmin(dim=1)
        has_candidate = candidate_pickups.any(dim=1)
        order_idx = torch.where(has_candidate, order_idx, torch.full_like(order_idx, -1))
        return order_idx

    def _classify_next_state_delivery_block(self, next_state):
        n_orders = next_state.n_orders
        if n_orders == 0:
            return 'other'

        ids_flat = next_state.ids.squeeze(-1)
        coords_active = next_state.coords[ids_flat]
        node_type_active = next_state.node_type[ids_flat]
        time_windows_active = next_state.time_windows[ids_flat]
        demand_p_full = next_state.demand_passenger[ids_flat]
        demand_c_full = next_state.demand_cargo[ids_flat]
        device = next_state.coords.device

        open_mask = next_state.get_open_started_mask()[0]
        open_indices = torch.nonzero(open_mask, as_tuple=False).squeeze(-1)
        if open_indices.numel() == 0:
            return 'other'

        delivery_indices = open_indices + n_orders + 1
        node_positions = torch.arange(coords_active.size(1), device=device)[None, :]
        pickup_not_done = (next_state.picked_up_ == 0).squeeze(1).bool()
        rejected = next_state.rejected_.squeeze(1).bool()
        precedence_mask = torch.zeros(1, 2 * n_orders, dtype=torch.bool, device=device)
        precedence_mask[:, :n_orders] = rejected
        precedence_mask[:, n_orders:] = pickup_not_done | rejected

        remaining_cap_p = next_state.PASSENGER_CAPACITY - next_state.used_capacity_passenger
        remaining_cap_c = next_state.CARGO_CAPACITY - next_state.used_capacity_cargo
        node_type_real = node_type_active[:, 1:2 * n_orders + 1]
        demand_p_real = demand_p_full[:, 1:2 * n_orders + 1]
        demand_c_real = demand_c_full[:, 1:2 * n_orders + 1]
        cap_mask_p = (node_type_real == 1) & (demand_p_real > remaining_cap_p + 1e-5)
        cap_mask_c = (node_type_real == 0) & (demand_c_real > remaining_cap_c + 1e-5)

        dist_to_nodes = (coords_active - next_state.cur_coord).norm(p=2, dim=-1) * next_state.AREA_SIZE
        travel_time = dist_to_nodes / next_state.VEHICLE_SPEED
        arrival_time = next_state.current_time + travel_time

        tw_end = time_windows_active[:, :, 1]
        is_passenger_node = node_type_active == 1
        is_pickup_node = (node_positions >= 1) & (node_positions <= n_orders)
        passenger_pickup_mask_full = (
            Config.HARD_PASSENGER_PICKUP_TIMEWINDOW
            & is_passenger_node
            & is_pickup_node
            & (arrival_time > tw_end + 1e-5)
        )

        order_indices = torch.arange(n_orders, device=device)
        delivery_node_indices = order_indices + n_orders + 1
        passenger_orders = (node_type_active[:, 1:n_orders + 1] == 1)
        delivery_arrival = arrival_time.gather(1, delivery_node_indices.unsqueeze(0))
        pickup_finish_time = next_state.passenger_pickup_time.squeeze(1)
        valid_pickup_time = pickup_finish_time >= 0
        ride_time_hours = delivery_arrival - pickup_finish_time
        direct_distance = torch.norm(
            coords_active[:, 1:n_orders + 1, :] - coords_active[:, n_orders + 1:2 * n_orders + 1, :],
            dim=-1
        ) * next_state.AREA_SIZE
        direct_ride_time_hours = direct_distance / next_state.VEHICLE_SPEED
        excess_ride_time_hours = torch.clamp(ride_time_hours - direct_ride_time_hours, min=0.0)
        passenger_total_ride_time_limit = Config.PASSENGER_MAX_RIDE_TIME_MINUTES / 60.0
        passenger_excess_ride_time_limit = Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES / 60.0
        ride_time_violation = (
            Config.HARD_PASSENGER_MAX_RIDE_TIME
            & passenger_orders
            & valid_pickup_time
            & (
                (ride_time_hours > passenger_total_ride_time_limit + 1e-5)
                | (excess_ride_time_hours > passenger_excess_ride_time_limit + 1e-5)
            )
        )

        depot_coord = coords_active[:, 0:1, :]
        dist_back = (coords_active - depot_coord).norm(p=2, dim=-1) * next_state.AREA_SIZE
        time_back = dist_back / next_state.VEHICLE_SPEED
        ongoing_trip_total = (arrival_time + next_state.SERVICE_TIME + time_back) - next_state.trip_start_time
        fresh_trip_total = travel_time + next_state.SERVICE_TIME + time_back
        predicted_total = torch.where(next_state.prev_a == 0, fresh_trip_total, ongoing_trip_total)
        trip_violation = Config.HARD_MAX_TRIP_TIME & (predicted_total > next_state.MAX_TRIP_TIME + 1e-5)
        predicted_finish = arrival_time + next_state.SERVICE_TIME + time_back
        ops_end_violation = Config.HARD_OPERATION_END & (predicted_finish > next_state.OPERATION_END + 1e-5)

        delivery_viability_mask = torch.zeros(n_orders, dtype=torch.bool, device=device)
        if n_orders > 0 and next_state.enable_delivery_viability:
            delivery_to_depot_time = (
                (coords_active[:, n_orders + 1:2 * n_orders + 1, :] - coords_active[:, 0:1, :]).norm(p=2, dim=-1)
                * next_state.AREA_SIZE / next_state.VEHICLE_SPEED
            )[0]
            delivery_earliest = time_windows_active[:, n_orders + 1:2 * n_orders + 1, 0][0]
            direct_ride_time = direct_distance[0] / next_state.VEHICLE_SPEED
            trip_start = next_state.trip_start_time[0, 0] if next_state.prev_a[0, 0].item() != 0 else next_state.current_time[0, 0]
            legal_orders, physical_orders, _ = next_state._get_legal_delivery_orders(
                next_state.cur_coord[0, 0],
                next_state.current_time[0, 0],
                trip_start,
                next_state.get_open_started_mask()[0],
                coords_active[0, n_orders + 1:2 * n_orders + 1, :],
                delivery_earliest,
                delivery_to_depot_time,
                passenger_orders[0],
                next_state.passenger_pickup_time.squeeze(1)[0],
                direct_ride_time,
                allow_fallback=next_state.enable_viability_fallback,
            )
            legal_set = set(legal_orders)
            for order_idx in physical_orders:
                if order_idx not in legal_set:
                    delivery_viability_mask[order_idx] = True

        vehicle_limit_mask = torch.zeros(n_orders, dtype=torch.bool, device=device)
        if Config.HARD_VEHICLE_LIMIT:
            import math as _math
            k_max = max(1, _math.ceil(n_orders * Config.DEFAULT_NUM_VEHICLE_RATIO))
            at_capacity = bool((next_state.used_vehicles >= k_max)[0, 0].item())
            at_depot = bool((next_state.prev_a == 0)[0, 0].item())
            all_done = bool((((next_state.visited_[:, :, 1:2 * n_orders + 1].sum(-1) == 2 * n_orders)
                              | (next_state.rejected_.sum(-1) == n_orders)).to(torch.uint8))[0, 0].item())
            if at_capacity and at_depot and (not all_done):
                vehicle_limit_mask[:] = True

        delivery_slice = delivery_indices - (n_orders + 1)
        precedence_block = precedence_mask[0, n_orders + delivery_slice]
        ride_block = ride_time_violation[0, delivery_slice]
        trip_block = trip_violation[0, delivery_indices]
        ops_block = ops_end_violation[0, delivery_indices]
        viability_block = delivery_viability_mask[delivery_slice]
        vehicle_limit_block = vehicle_limit_mask[delivery_slice]

        reason_masks = {
            'precedence': precedence_block,
            'ride_time': ride_block,
            'trip_time': trip_block,
            'ops_end': ops_block,
            'delivery_viability': viability_block,
            'vehicle_limit': vehicle_limit_block,
        }

        covering = [name for name, mask in reason_masks.items() if bool(mask.all().item())]
        if len(covering) == 1:
            return covering[0]
        if len(covering) > 1:
            return 'mixed'

        combined = torch.zeros_like(precedence_block)
        for mask in reason_masks.values():
            combined |= mask
        if bool(combined.all().item()):
            return 'mixed'
        return 'other'

    def get_mask(self, return_debug=False, skip_pickup_commitment=False):
        batch_size = self.ids.size(0)
        n_orders = self.n_orders
        device = self.coords.device

        ids_flat, coords_active, node_type_active, time_windows_active, demand_p_full, demand_c_full = self._active_views()

        debug = None
        if return_debug:
            debug_keys = [
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
            ]
            debug = {key: torch.zeros(batch_size, device=device) for key in debug_keys}

        service_mask = self.visited[:, :, 1:2 * n_orders + 1].bool().squeeze(1).clone()
        if return_debug:
            debug['diag_mask_visited'] = service_mask.sum(1).float()

        pickup_not_done = (self.picked_up_ == 0).squeeze(1).bool()
        rejected = self.rejected_.squeeze(1).bool()
        precedence_mask = torch.zeros_like(service_mask)
        if n_orders > 0:
            precedence_mask[:, :n_orders] = rejected
            precedence_mask[:, n_orders:] = pickup_not_done | rejected
        before_mask = service_mask.clone()
        service_mask |= precedence_mask
        if return_debug:
            debug['diag_mask_precedence'] = (service_mask & ~before_mask).sum(1).float()

        remaining_cap_p = self.PASSENGER_CAPACITY - self.used_capacity_passenger
        remaining_cap_c = self.CARGO_CAPACITY - self.used_capacity_cargo
        node_type_real = node_type_active[:, 1:2 * n_orders + 1]
        demand_p_real = demand_p_full[:, 1:2 * n_orders + 1]
        demand_c_real = demand_c_full[:, 1:2 * n_orders + 1]

        cap_mask_p = (node_type_real == 1) & (demand_p_real > remaining_cap_p + 1e-5)
        before_mask = service_mask.clone()
        service_mask |= cap_mask_p
        if return_debug:
            debug['diag_mask_cap_passenger'] = (service_mask & ~before_mask).sum(1).float()

        cap_mask_c = (node_type_real == 0) & (demand_c_real > remaining_cap_c + 1e-5)
        before_mask = service_mask.clone()
        service_mask |= cap_mask_c
        if return_debug:
            debug['diag_mask_cap_cargo'] = (service_mask & ~before_mask).sum(1).float()

        dist_to_nodes = (coords_active - self.cur_coord).norm(p=2, dim=-1) * self.AREA_SIZE
        travel_time = dist_to_nodes / self.VEHICLE_SPEED
        arrival_time = self.current_time + travel_time

        tw_end = time_windows_active[:, :, 1]
        node_positions = torch.arange(coords_active.size(1), device=device)[None, :]
        is_passenger_node = node_type_active == 1
        is_pickup_node = (node_positions >= 1) & (node_positions <= n_orders)

        passenger_pickup_mask_full = (
            Config.HARD_PASSENGER_PICKUP_TIMEWINDOW
            & is_passenger_node
            & is_pickup_node
            & (arrival_time > tw_end + 1e-5)
        )
        before_mask = service_mask.clone()
        service_mask |= passenger_pickup_mask_full[:, 1:2 * n_orders + 1]
        if return_debug:
            debug['diag_mask_pickup_tw'] = (service_mask & ~before_mask).sum(1).float()

        open_started_mask = self.get_open_started_mask()
        open_started_count = self.get_open_started_count().squeeze(1)
        open_started = open_started_count > 0
        if return_debug:
            debug['diag_open_started_count'] = open_started_count.float()
            debug['diag_open_started_eq2'] = (open_started_count == 2).float()

        if n_orders > 0:
            order_indices = torch.arange(n_orders, device=device)
            delivery_indices = order_indices + n_orders + 1
            passenger_orders = (node_type_active[:, 1:n_orders + 1] == 1)
            delivery_arrival = arrival_time.gather(1, delivery_indices.unsqueeze(0).expand(batch_size, -1))
            pickup_finish_time = self.passenger_pickup_time.squeeze(1)
            valid_pickup_time = pickup_finish_time >= 0
            ride_time_hours = delivery_arrival - pickup_finish_time
            direct_distance = torch.norm(
                coords_active[:, 1:n_orders + 1, :] - coords_active[:, n_orders + 1:2 * n_orders + 1, :],
                dim=-1
            ) * self.AREA_SIZE
            direct_ride_time_hours = direct_distance / self.VEHICLE_SPEED
            excess_ride_time_hours = torch.clamp(ride_time_hours - direct_ride_time_hours, min=0.0)
            passenger_total_ride_time_limit = Config.PASSENGER_MAX_RIDE_TIME_MINUTES / 60.0
            passenger_excess_ride_time_limit = Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES / 60.0
            ride_time_violation = (
                Config.HARD_PASSENGER_MAX_RIDE_TIME
                & passenger_orders
                & valid_pickup_time
                & (
                    (ride_time_hours > passenger_total_ride_time_limit + 1e-5)
                    | (excess_ride_time_hours > passenger_excess_ride_time_limit + 1e-5)
                )
            )
            ride_time_mask_full = torch.zeros(batch_size, coords_active.size(1), dtype=torch.bool, device=device)
            ride_time_mask_full.scatter_(1, delivery_indices.unsqueeze(0).expand(batch_size, -1), ride_time_violation)
        else:
            ride_time_mask_full = torch.zeros(batch_size, coords_active.size(1), dtype=torch.bool, device=device)
        before_mask = service_mask.clone()
        service_mask |= ride_time_mask_full[:, 1:2 * n_orders + 1]
        if return_debug:
            debug['diag_mask_ride_time'] = (service_mask & ~before_mask).sum(1).float()

        trip_mask_full = torch.zeros(batch_size, coords_active.size(1), dtype=torch.bool, device=device)
        if Config.HARD_MAX_TRIP_TIME:
            depot_coord = coords_active[:, 0:1, :]
            dist_back = (coords_active - depot_coord).norm(p=2, dim=-1) * self.AREA_SIZE
            time_back = dist_back / self.VEHICLE_SPEED
            ongoing_trip_total = (arrival_time + self.SERVICE_TIME + time_back) - self.trip_start_time
            fresh_trip_total = travel_time + self.SERVICE_TIME + time_back
            predicted_total = torch.where(self.prev_a == 0, fresh_trip_total, ongoing_trip_total)
            trip_mask_full = predicted_total > self.MAX_TRIP_TIME + 1e-5
            trip_mask_full[:, 0] = False
        before_mask = service_mask.clone()
        service_mask |= trip_mask_full[:, 1:2 * n_orders + 1]
        if return_debug:
            debug['diag_mask_trip_time'] = (service_mask & ~before_mask).sum(1).float()

        ops_end_mask_full = torch.zeros(batch_size, coords_active.size(1), dtype=torch.bool, device=device)
        if Config.HARD_OPERATION_END:
            depot_coord_v6 = coords_active[:, 0:1, :]
            dist_back_v6 = (coords_active - depot_coord_v6).norm(p=2, dim=-1) * self.AREA_SIZE
            time_back_v6 = dist_back_v6 / self.VEHICLE_SPEED
            predicted_finish = arrival_time + self.SERVICE_TIME + time_back_v6
            ops_end_mask_full = predicted_finish > self.OPERATION_END + 1e-5
            ops_end_mask_full[:, 0] = False
        before_mask = service_mask.clone()
        service_mask |= ops_end_mask_full[:, 1:2 * n_orders + 1]
        if return_debug:
            debug['diag_mask_ops_end'] = (service_mask & ~before_mask).sum(1).float()

        pickup_commitment_mask = torch.zeros(batch_size, n_orders, dtype=torch.bool, device=device)
        if n_orders > 0 and (not skip_pickup_commitment):
            pickup_coords = coords_active[:, 1:n_orders + 1, :]
            delivery_coords = coords_active[:, n_orders + 1:2 * n_orders + 1, :]
            delivery_earliest = time_windows_active[:, n_orders + 1:2 * n_orders + 1, 0]
            pickup_finish = torch.maximum(
                arrival_time[:, 1:n_orders + 1],
                time_windows_active[:, 1:n_orders + 1, 0]
            ) + self.SERVICE_TIME
            open_before = open_started_mask
            delivery_to_depot_time = (
                (delivery_coords - coords_active[:, 0:1, :]).norm(p=2, dim=-1) * self.AREA_SIZE / self.VEHICLE_SPEED
            )
            trip_start_after_pickup = torch.where(self.prev_a == 0, self.current_time, self.trip_start_time).squeeze(1)
            passenger_orders = (node_type_active[:, 1:n_orders + 1] == 1)
            passenger_pickup_times_before = self.passenger_pickup_time.squeeze(1)
            direct_ride_time = (
                (pickup_coords - delivery_coords).norm(p=2, dim=-1) * self.AREA_SIZE / self.VEHICLE_SPEED
            )
            pickup_physical_feasible = ~service_mask[:, :n_orders]

            for batch_idx in range(batch_size):
                candidate_indices = torch.nonzero(pickup_physical_feasible[batch_idx], as_tuple=False).squeeze(-1)
                if candidate_indices.numel() == 0:
                    continue
                second_pickup_feasible = 0
                second_pickup_blocked = 0
                if open_started_count[batch_idx].item() >= self.max_concurrent_open_orders:
                    pickup_commitment_mask[batch_idx, candidate_indices] = True
                    second_pickup_blocked = int(candidate_indices.numel())
                    if return_debug:
                        debug['diag_pickup_commitment_block_by_k'][batch_idx] += float(candidate_indices.numel())
                else:
                    batch_state = self[batch_idx:batch_idx + 1]
                    batch_mask = batch_state.get_mask(skip_pickup_commitment=True)
                    for candidate_idx_tensor in candidate_indices:
                        candidate_idx = int(candidate_idx_tensor.item())
                        open_after = open_before[batch_idx].clone()
                        open_after[candidate_idx] = True
                        if open_after.sum().item() > self.max_concurrent_open_orders:
                            pickup_commitment_mask[batch_idx, candidate_idx] = True
                            second_pickup_blocked += 1
                            if return_debug:
                                debug['diag_pickup_commitment_block_by_k'][batch_idx] += 1.0
                            continue

                        passenger_pickup_times_after = passenger_pickup_times_before[batch_idx].clone()
                        if passenger_orders[batch_idx, candidate_idx].item():
                            passenger_pickup_times_after[candidate_idx] = pickup_finish[batch_idx, candidate_idx]
                        completion_feasible, completion_block_reason = self._has_feasible_open_completion(
                            pickup_coords[batch_idx, candidate_idx],
                            pickup_finish[batch_idx, candidate_idx],
                            trip_start_after_pickup[batch_idx],
                            open_after,
                            delivery_coords[batch_idx],
                            delivery_earliest[batch_idx],
                            delivery_to_depot_time[batch_idx],
                            passenger_orders[batch_idx],
                            passenger_pickup_times_after,
                            direct_ride_time[batch_idx],
                            return_reason=return_debug,
                            ignore_trip_time=self.relax_pickup_commitment_trip_time,
                        ) if return_debug else (
                            self._has_feasible_open_completion(
                                pickup_coords[batch_idx, candidate_idx],
                                pickup_finish[batch_idx, candidate_idx],
                                trip_start_after_pickup[batch_idx],
                                open_after,
                                delivery_coords[batch_idx],
                                delivery_earliest[batch_idx],
                                delivery_to_depot_time[batch_idx],
                                passenger_orders[batch_idx],
                                passenger_pickup_times_after,
                                direct_ride_time[batch_idx],
                                ignore_trip_time=self.relax_pickup_commitment_trip_time,
                            ),
                            None,
                        )

                        candidate_node = candidate_idx + 1
                        selected = torch.tensor([candidate_node], dtype=torch.long, device=device)
                        next_state = batch_state.update(selected, current_mask=batch_mask)
                        next_mask = next_state.get_mask(skip_pickup_commitment=True)
                        next_open_count = int(next_state.get_open_started_count()[0, 0].item())
                        next_feasible_deliveries = int((~next_mask[0, 0, n_orders + 1:2 * n_orders + 1]).sum().item())
                        next_state_feasible = (next_open_count == 0) or (next_feasible_deliveries > 0)

                        fallback_safe = True
                        if self.enable_viability_fallback and next_open_count > 0:
                            next_delivery_coords = next_state.coords[:, n_orders + 1:2 * n_orders + 1, :][0]
                            next_delivery_earliest = next_state.time_windows[:, n_orders + 1:2 * n_orders + 1, 0][0]
                            next_delivery_to_depot_time = (
                                (next_delivery_coords - next_state.coords[:, 0:1, :][0]).norm(p=2, dim=-1)
                                * self.AREA_SIZE / self.VEHICLE_SPEED
                            )
                            next_passenger_orders = (next_state.node_type[:, 1:n_orders + 1] == 1)[0]
                            next_passenger_pickup_times = next_state.passenger_pickup_time.squeeze(1)[0]
                            next_direct_ride_time = (
                                (next_state.coords[:, 1:n_orders + 1, :][0] - next_delivery_coords).norm(p=2, dim=-1)
                                * self.AREA_SIZE / self.VEHICLE_SPEED
                            )
                            next_trip_start = next_state.trip_start_time[0, 0] if next_state.prev_a[0, 0].item() != 0 else next_state.current_time[0, 0]
                            fallback_safe = self._has_any_physical_delivery_step(
                                next_state.cur_coord[0, 0],
                                next_state.current_time[0, 0],
                                next_trip_start,
                                next_state.get_open_started_mask()[0],
                                next_delivery_coords,
                                next_delivery_earliest,
                                next_delivery_to_depot_time,
                                next_passenger_orders,
                                next_passenger_pickup_times,
                                next_direct_ride_time,
                            )

                        candidate_feasible = completion_feasible and next_state_feasible and fallback_safe
                        pickup_commitment_mask[batch_idx, candidate_idx] = not candidate_feasible
                        if candidate_feasible:
                            second_pickup_feasible += 1
                        else:
                            second_pickup_blocked += 1
                            if return_debug:
                                if not completion_feasible:
                                    debug['diag_pickup_commitment_block_by_completion'][batch_idx] += 1.0
                                    if completion_block_reason == 'ride_time':
                                        debug['diag_pickup_commitment_block_by_completion_ride_time'][batch_idx] += 1.0
                                    elif completion_block_reason == 'trip_time':
                                        debug['diag_pickup_commitment_block_by_completion_trip_time'][batch_idx] += 1.0
                                    elif completion_block_reason == 'ops_end':
                                        debug['diag_pickup_commitment_block_by_completion_ops_end'][batch_idx] += 1.0
                                    elif completion_block_reason == 'open_over_6':
                                        debug['diag_pickup_commitment_block_by_completion_open_over_6'][batch_idx] += 1.0
                                    else:
                                        debug['diag_pickup_commitment_block_by_completion_other'][batch_idx] += 1.0
                                elif not next_state_feasible:
                                    debug['diag_pickup_commitment_block_by_next_state'][batch_idx] += 1.0
                                    next_state_reason = self._classify_next_state_delivery_block(next_state)
                                    if next_state_reason == 'precedence':
                                        debug['diag_pickup_commitment_block_by_next_state_precedence'][batch_idx] += 1.0
                                    elif next_state_reason == 'ride_time':
                                        debug['diag_pickup_commitment_block_by_next_state_ride_time'][batch_idx] += 1.0
                                    elif next_state_reason == 'trip_time':
                                        debug['diag_pickup_commitment_block_by_next_state_trip_time'][batch_idx] += 1.0
                                    elif next_state_reason == 'ops_end':
                                        debug['diag_pickup_commitment_block_by_next_state_ops_end'][batch_idx] += 1.0
                                    elif next_state_reason == 'delivery_viability':
                                        debug['diag_pickup_commitment_block_by_next_state_delivery_viability'][batch_idx] += 1.0
                                    elif next_state_reason == 'vehicle_limit':
                                        debug['diag_pickup_commitment_block_by_next_state_vehicle_limit'][batch_idx] += 1.0
                                    elif next_state_reason == 'mixed':
                                        debug['diag_pickup_commitment_block_by_next_state_mixed'][batch_idx] += 1.0
                                    else:
                                        debug['diag_pickup_commitment_block_by_next_state_other'][batch_idx] += 1.0
                                elif not fallback_safe:
                                    debug['diag_pickup_commitment_block_by_fallback'][batch_idx] += 1.0
                if return_debug and open_started_count[batch_idx].item() >= 1:
                    debug['diag_second_pickup_feasible'][batch_idx] = float(second_pickup_feasible)
                    debug['diag_second_pickup_blocked_by_commitment'][batch_idx] = float(second_pickup_blocked)

        commitment_mask_full = torch.zeros(batch_size, coords_active.size(1), dtype=torch.bool, device=device)
        if n_orders > 0:
            commitment_mask_full[:, 1:n_orders + 1] = pickup_commitment_mask
        before_mask = service_mask.clone()
        service_mask |= commitment_mask_full[:, 1:2 * n_orders + 1]
        if return_debug:
            debug['diag_mask_pickup_commitment'] = (service_mask & ~before_mask).sum(1).float()

        if n_orders > 0 and self.enable_delivery_viability:
            delivery_viability_mask = torch.zeros(batch_size, n_orders, dtype=torch.bool, device=device)
            delivery_to_depot_time = (
                (coords_active[:, n_orders + 1:2 * n_orders + 1, :] - coords_active[:, 0:1, :]).norm(p=2, dim=-1)
                * self.AREA_SIZE / self.VEHICLE_SPEED
            )
            delivery_earliest = time_windows_active[:, n_orders + 1:2 * n_orders + 1, 0]
            direct_ride_time = direct_distance / self.VEHICLE_SPEED
            open_mask_full = self.get_open_started_mask()
            for batch_idx in range(batch_size):
                trip_start = self.trip_start_time[batch_idx, 0] if self.prev_a[batch_idx, 0].item() != 0 else self.current_time[batch_idx, 0]
                legal_orders, physical_orders, used_fallback = self._get_legal_delivery_orders(
                    self.cur_coord[batch_idx, 0],
                    self.current_time[batch_idx, 0],
                    trip_start,
                    open_mask_full[batch_idx],
                    coords_active[batch_idx, n_orders + 1:2 * n_orders + 1, :],
                    delivery_earliest[batch_idx],
                    delivery_to_depot_time[batch_idx],
                    passenger_orders[batch_idx],
                    self.passenger_pickup_time.squeeze(1)[batch_idx],
                    direct_ride_time[batch_idx],
                    allow_fallback=self.enable_viability_fallback,
                )
                if len(physical_orders) == 0:
                    continue
                legal_set = set(legal_orders)
                for order_idx in physical_orders:
                    if order_idx not in legal_set:
                        delivery_viability_mask[batch_idx, order_idx] = True
                if used_fallback and return_debug:
                    debug['diag_delivery_viability_fallback'][batch_idx] = 1.0
            delivery_viability_mask_full = torch.zeros(batch_size, coords_active.size(1), dtype=torch.bool, device=device)
            delivery_viability_mask_full[:, n_orders + 1:2 * n_orders + 1] = delivery_viability_mask
            before_viability = service_mask.clone()
            service_mask |= delivery_viability_mask_full[:, 1:2 * n_orders + 1]
            if return_debug:
                debug['diag_delivery_viability_masked'] = (service_mask & ~before_viability).sum(1).float()

        mask = self.visited.clone()
        if n_orders > 0:
            mask[:, :, 1:2 * n_orders + 1] = service_mask.unsqueeze(1).to(torch.uint8)
        reject_index = self.reject_index

        is_carrying = ((self.used_capacity_passenger > 1e-5) | (self.used_capacity_cargo > 1e-5)).squeeze(1)
        if return_debug:
            debug['diag_depot_carry_block'] = is_carrying.float()
        mask[:, :, 0] = mask[:, :, 0] | is_carrying[:, None].to(torch.uint8)

        n_visited = self.visited_[:, :, 1:2 * n_orders + 1].sum(-1)
        at_depot_no_work = ((self.prev_a == 0) & (n_visited == 0)).squeeze(1)
        if return_debug:
            debug['diag_depot_no_work_block'] = at_depot_no_work.float()
        mask[:, :, 0] = mask[:, :, 0] | at_depot_no_work[:, None].to(torch.uint8)

        all_done = ((self.visited_[:, :, 1:2 * n_orders + 1].sum(-1) == 2 * n_orders) | (self.rejected_.sum(-1) == n_orders)).to(torch.uint8)
        if Config.HARD_VEHICLE_LIMIT:
            import math as _math
            k_max = max(1, _math.ceil(n_orders * Config.DEFAULT_NUM_VEHICLE_RATIO))
            at_capacity = (self.used_vehicles >= k_max).to(torch.uint8)
            at_depot = (self.prev_a == 0).to(torch.uint8)
            block_all_when_exhausted = at_capacity & at_depot & (1 - all_done)
            if block_all_when_exhausted.any():
                before_service = mask[:, :, 1:2 * n_orders + 1].bool().squeeze(1).clone()
                expanded = block_all_when_exhausted[:, :, None].expand_as(mask[:, :, 1:2 * n_orders + 1]).bool()
                mask[:, :, 1:2 * n_orders + 1] = mask[:, :, 1:2 * n_orders + 1] | expanded.to(torch.uint8)
                if return_debug:
                    after_service = mask[:, :, 1:2 * n_orders + 1].bool().squeeze(1)
                    debug['diag_mask_vehicle_limit'] = (after_service & ~before_service).sum(1).float()
                mask[:, :, 0] = mask[:, :, 0] | block_all_when_exhausted
        
        all_order_nodes_masked = mask[:, :, 1:2 * n_orders + 1].bool().all(-1)
        if Config.HARD_VEHICLE_LIMIT:
            import math as _math
            k_max = max(1, _math.ceil(n_orders * Config.DEFAULT_NUM_VEHICLE_RATIO))
            can_start_new_vehicle = (self.used_vehicles < k_max) & (self.prev_a == 0)
            allow_depot_fallback = all_order_nodes_masked & can_start_new_vehicle
        else:
            allow_depot_fallback = all_order_nodes_masked
        if return_debug:
            debug['diag_depot_fallback_used'] = allow_depot_fallback.squeeze(-1).float()
        mask[:, :, 0] = mask[:, :, 0] & (~allow_depot_fallback).to(torch.uint8)

        reject_candidates = self._deterministic_reject_order(mask)
        reject_candidate_available = reject_candidates >= 0
        pre_departure_gate = (
            (self.prev_a == 0)
            & (self.used_capacity_passenger <= 1e-5)
            & (self.used_capacity_cargo <= 1e-5)
            & (~self.has_open_started_orders())
        ).squeeze(1)
        reject_allowed = (
            reject_candidate_available
            & (~all_done.squeeze(-1).bool())
            & bool(self.allow_reject)
            & pre_departure_gate
        )
        mask[:, :, reject_index] = (~reject_allowed).view(-1, 1).to(torch.uint8)
        if return_debug:
            reject_available = mask[:, :, reject_index].eq(0).view(-1)
            debug['diag_reject_candidate_available'] = reject_candidate_available.float()
            debug['diag_reject_allowed'] = reject_allowed.float()
            debug['diag_reject_predeparture_available'] = (reject_available & pre_departure_gate).float()
            debug['diag_reject_inroute_available'] = (reject_available & (~pre_departure_gate)).float()

        if return_debug:
            return mask.bool(), debug
        return mask.bool()

    def update(self, selected, current_mask=None):
        selected = selected[:, None]
        n_orders = self.n_orders
        DEPOT = 0
        reject_index = self.reject_index

        ids_flat, coords_active, node_type_active, time_windows_active, demand_p_active, demand_c_active = self._active_views()
        if current_mask is None:
            current_mask = self.get_mask(skip_pickup_commitment=True)
        reject_targets = self._deterministic_reject_order(current_mask)

        is_reject = selected == reject_index
        actual_selected = selected.clone()
        safe_targets = torch.clamp(reject_targets + 1, min=0)
        actual_selected[is_reject] = safe_targets[:, None][is_reject]

        selected_coord = coords_active.gather(
            1, actual_selected[:, :, None].expand(-1, -1, 2)
        )

        dist = (selected_coord - self.cur_coord).norm(p=2, dim=-1) * self.AREA_SIZE
        travel_time = dist / self.VEHICLE_SPEED
        dist = torch.where(is_reject, torch.zeros_like(dist), dist)
        travel_time = torch.where(is_reject, torch.zeros_like(travel_time), travel_time)
        selected_coord = torch.where(is_reject[:, :, None], self.cur_coord, selected_coord)

        new_lengths = self.lengths + dist
        arrival_time = self.current_time + travel_time

        is_depot = (actual_selected == DEPOT).float()
        leaving_depot = (self.prev_a == DEPOT) & (actual_selected != DEPOT) & (~is_reject)

        new_time = torch.where(is_reject, self.current_time, is_depot * self.OPERATION_START + (1 - is_depot) * arrival_time)
        dispatch_time = torch.where(
            is_reject,
            self.current_time,
            torch.where(
                leaving_depot,
                torch.maximum(self.current_time, time_windows_active.gather(1, actual_selected[:, :, None].expand(-1, -1, 2))[:, :, 0] - travel_time),
                self.trip_start_time,
            ),
        )
        new_used_vehicles = self.used_vehicles + leaving_depot.float()

        tw = time_windows_active.gather(1, actual_selected[:, :, None].expand(-1, -1, 2))
        earliest = tw[:, :, 0]
        start_service = torch.max(new_time, earliest)
        new_time = torch.where(is_reject, self.current_time, (1 - is_depot) * (start_service + self.SERVICE_TIME) + is_depot * new_time)

        demand_p = demand_p_active.gather(1, actual_selected)
        demand_c = demand_c_active.gather(1, actual_selected)
        demand_p = torch.where(is_reject, torch.zeros_like(demand_p), demand_p)
        demand_c = torch.where(is_reject, torch.zeros_like(demand_c), demand_c)

        new_cap_p = (1 - is_depot) * (self.used_capacity_passenger + demand_p)
        new_cap_c = (1 - is_depot) * (self.used_capacity_cargo + demand_c)
        new_cap_p = torch.where(is_reject, self.used_capacity_passenger, torch.clamp(new_cap_p, min=0.0))
        new_cap_c = torch.where(is_reject, self.used_capacity_cargo, torch.clamp(new_cap_c, min=0.0))

        new_trip_start = torch.where(
            is_reject,
            self.trip_start_time,
            torch.where(leaving_depot, dispatch_time, torch.where(is_depot > 0, new_time, self.trip_start_time))
        )

        all_order_nodes_masked = current_mask[:, :, 1:2 * n_orders + 1].all(-1).to(torch.long)
        new_deadlock_count = torch.where(
            (is_depot > 0) & (~is_reject),
            self.deadlock_count + all_order_nodes_masked,
            torch.zeros_like(self.deadlock_count),
        )
        deadlock_limit = torch.clamp(self.deadlock_limit, min=1)
        new_terminal = self.terminal_ | (new_deadlock_count >= deadlock_limit)

        visited_ = self.visited_.clone()
        non_reject_selected = actual_selected.unsqueeze(-1)
        visited_ = visited_.scatter(-1, non_reject_selected, torch.where(is_reject.unsqueeze(-1), torch.zeros_like(non_reject_selected, dtype=visited_.dtype), torch.ones_like(non_reject_selected, dtype=visited_.dtype)))
        visited_[:, :, DEPOT] = 0
        visited_[:, :, reject_index] = 0

        picked_up_ = self.picked_up_.clone()
        passenger_pickup_time = self.passenger_pickup_time.clone()
        rejected_ = self.rejected_.clone()
        reject_count = self.reject_count.clone()

        is_pickup = (actual_selected >= 1) & (actual_selected <= n_orders)
        pickup_order_idx = (actual_selected - 1).clamp(min=0, max=max(n_orders - 1, 0))

        if n_orders > 0:
            pickup_mask = is_pickup.unsqueeze(-1).expand(-1, -1, n_orders)
            order_one_hot = torch.zeros_like(picked_up_)
            order_one_hot.scatter_(-1, pickup_order_idx.unsqueeze(-1), 1)
            picked_up_ = picked_up_ | ((pickup_mask & order_one_hot.bool()) & (~is_reject.unsqueeze(-1))).to(torch.uint8)

            passenger_selected = (node_type_active.gather(1, actual_selected) == 1)
            passenger_pickup_selected = is_pickup & passenger_selected & (~is_reject)
            time_update = torch.zeros_like(passenger_pickup_time)
            time_update.scatter_(-1, pickup_order_idx.unsqueeze(-1), new_time.unsqueeze(-1))
            update_mask = (pickup_mask & passenger_pickup_selected.unsqueeze(-1)).to(passenger_pickup_time.dtype)
            passenger_pickup_time = passenger_pickup_time * (1.0 - update_mask) + time_update * update_mask

            reject_valid = reject_targets >= 0
            if reject_valid.any():
                reject_target_idx = torch.clamp(reject_targets, min=0, max=n_orders - 1)
                reject_target_view = reject_target_idx.view(-1, 1, 1)
                reject_one_hot = torch.zeros_like(rejected_)
                reject_one_hot.scatter_(-1, reject_target_view, 1)
                reject_mask = is_reject.unsqueeze(-1) & reject_valid.view(-1, 1, 1)
                rejected_ = rejected_ | (reject_one_hot & reject_mask).to(torch.uint8)
                visited_[:, :, 1:n_orders + 1] = visited_[:, :, 1:n_orders + 1] | (reject_one_hot & reject_mask).to(torch.uint8)
                visited_[:, :, n_orders + 1:2 * n_orders + 1] = visited_[:, :, n_orders + 1:2 * n_orders + 1] | (reject_one_hot & reject_mask).to(torch.uint8)
                reject_count = reject_count + is_reject.float() * reject_valid.float().view(-1, 1)

        new_prev_a = torch.where(is_reject, self.prev_a, actual_selected)

        return self._replace(
            current_time=new_time,
            trip_start_time=new_trip_start,
            prev_a=new_prev_a,
            used_capacity_passenger=new_cap_p,
            used_capacity_cargo=new_cap_c,
            used_vehicles=new_used_vehicles,
            visited_=visited_,
            picked_up_=picked_up_,
            passenger_pickup_time=passenger_pickup_time,
            rejected_=rejected_,
            reject_count=reject_count,
            lengths=new_lengths,
            cur_coord=selected_coord,
            deadlock_count=new_deadlock_count,
            deadlock_limit=self.deadlock_limit,
            terminal_=new_terminal,
            i=self.i + 1,
        )

    def all_finished(self):
        open_started = self.has_open_started_orders()
        all_visited = ((self.visited_[:, :, 1:2 * self.n_orders + 1].sum(-1) == self.n_orders * 2) | (self.rejected_.sum(-1) == self.n_orders))
        at_depot = self.prev_a == 0
        return ((((all_visited & at_depot) | self.terminal_) & (~open_started))).all()

    def get_finished(self):
        open_started = self.has_open_started_orders()
        all_visited = ((self.visited_[:, :, 1:2 * self.n_orders + 1].sum(-1) == self.n_orders * 2) | (self.rejected_.sum(-1) == self.n_orders))
        at_depot = self.prev_a == 0
        deadlock_finished = self.deadlock_count >= torch.clamp(self.deadlock_limit, min=1)
        return ((((all_visited & at_depot) | deadlock_finished | self.terminal_) & (~open_started))).squeeze(-1)

    def get_current_node(self):
        return self.prev_a

    def get_remaining_vehicle_budget(self):
        import math as _math
        k_max = max(1, _math.ceil(self.n_orders * Config.DEFAULT_NUM_VEHICLE_RATIO))
        remaining = torch.clamp(k_max - self.used_vehicles, min=0.0)
        return remaining / float(k_max)

    def get_final_cost(self):
        coords_active = self.coords[self.ids.squeeze(-1)]
        depot_coord = coords_active[:, 0:1, :]
        return_dist = (self.cur_coord - depot_coord).norm(p=2, dim=-1) * self.AREA_SIZE
        return self.lengths + return_dist

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return self._replace(
                ids=self.ids[key],
                current_time=self.current_time[key],
                trip_start_time=self.trip_start_time[key],
                prev_a=self.prev_a[key],
                used_capacity_passenger=self.used_capacity_passenger[key],
                used_capacity_cargo=self.used_capacity_cargo[key],
                used_vehicles=self.used_vehicles[key],
                visited_=self.visited_[key],
                picked_up_=self.picked_up_[key],
                passenger_pickup_time=self.passenger_pickup_time[key],
                rejected_=self.rejected_[key],
                reject_count=self.reject_count[key],
                allow_reject=self.allow_reject,
                max_concurrent_open_orders=self.max_concurrent_open_orders,
                enable_delivery_viability=self.enable_delivery_viability,
                enable_viability_fallback=self.enable_viability_fallback,
                relax_pickup_commitment_trip_time=self.relax_pickup_commitment_trip_time,
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                deadlock_count=self.deadlock_count[key],
                deadlock_limit=self.deadlock_limit[key],
                terminal_=self.terminal_[key],
                i=self.i[key],
            )
        raise TypeError(f"Invalid key type: {type(key)}")
