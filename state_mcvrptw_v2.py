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
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    deadlock_count: torch.Tensor
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

    @staticmethod
    def initialize(input_data, visited_dtype=torch.uint8):
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
            lengths=torch.zeros(batch_size, 1, device=device),
            cur_coord=depot[:, None, :],
            deadlock_count=torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            terminal_=torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            i=torch.zeros(1, dtype=torch.int64, device=device),
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

    def _deterministic_reject_order(self, mask):
        batch_size = mask.size(0)
        device = mask.device
        n_orders = self.n_orders
        if n_orders == 0:
            return torch.full((batch_size,), -1, dtype=torch.long, device=device)

        candidate_pickups = ~mask[:, 0, 1:n_orders + 1].bool()
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

    def get_mask(self):
        batch_size = self.ids.size(0)
        n_orders = self.n_orders
        device = self.coords.device

        ids_flat, coords_active, node_type_active, time_windows_active, demand_p_full, demand_c_full = self._active_views()

        mask = self.visited.clone()
        reject_index = self.reject_index

        pickup_not_done = (self.picked_up_ == 0).to(torch.uint8)
        rejected = self.rejected_.to(torch.uint8)
        mask[:, :, n_orders + 1: 2 * n_orders + 1] = (
            mask[:, :, n_orders + 1: 2 * n_orders + 1] | pickup_not_done | rejected
        )
        mask[:, :, 1:n_orders + 1] = mask[:, :, 1:n_orders + 1] | rejected

        remaining_cap_p = self.PASSENGER_CAPACITY - self.used_capacity_passenger
        remaining_cap_c = self.CARGO_CAPACITY - self.used_capacity_cargo

        is_passenger = node_type_active == 1
        cap_mask_p = is_passenger & (demand_p_full > remaining_cap_p + 1e-5)

        is_cargo = node_type_active == 0
        cap_mask_c = is_cargo & (demand_c_full > remaining_cap_c + 1e-5)

        mask[:, :, :2 * n_orders + 1] = mask[:, :, :2 * n_orders + 1] | cap_mask_p.unsqueeze(1).to(torch.uint8) | cap_mask_c.unsqueeze(1).to(torch.uint8)

        dist_to_nodes = (coords_active - self.cur_coord).norm(p=2, dim=-1) * self.AREA_SIZE
        travel_time = dist_to_nodes / self.VEHICLE_SPEED
        arrival_time = self.current_time + travel_time

        tw_end = time_windows_active[:, :, 1]
        node_positions = torch.arange(coords_active.size(1), device=device)[None, :]
        is_passenger_node = node_type_active == 1
        is_pickup_node = (node_positions >= 1) & (node_positions <= n_orders)
        is_delivery_node = (node_positions >= n_orders + 1) & (node_positions <= 2 * n_orders)

        passenger_pickup_mask = (
            Config.HARD_PASSENGER_PICKUP_TIMEWINDOW
            & is_passenger_node
            & is_pickup_node
            & (arrival_time > tw_end + 1e-5)
        )
        mask[:, :, :2 * n_orders + 1] = mask[:, :, :2 * n_orders + 1] | passenger_pickup_mask.unsqueeze(1).to(torch.uint8)

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
            ride_time_mask = torch.zeros(batch_size, coords_active.size(1), dtype=torch.bool, device=device)
            ride_time_mask.scatter_(1, delivery_indices.unsqueeze(0).expand(batch_size, -1), ride_time_violation)
            mask[:, :, :2 * n_orders + 1] = mask[:, :, :2 * n_orders + 1] | ride_time_mask.unsqueeze(1).to(torch.uint8)

        if Config.HARD_MAX_TRIP_TIME:
            depot_coord = coords_active[:, 0:1, :]
            dist_back = (coords_active - depot_coord).norm(p=2, dim=-1) * self.AREA_SIZE
            time_back = dist_back / self.VEHICLE_SPEED
            predicted_total = (arrival_time + self.SERVICE_TIME + time_back) - self.trip_start_time
            if predicted_total.dim() == 2:
                predicted_total = predicted_total.unsqueeze(1)
            trip_mask = (predicted_total > self.MAX_TRIP_TIME + 1e-5)
            trip_mask[:, :, 0] = False
            mask[:, :, :2 * n_orders + 1] = mask[:, :, :2 * n_orders + 1] | trip_mask.to(torch.uint8)

        if Config.HARD_OPERATION_END:
            depot_coord_v6 = coords_active[:, 0:1, :]
            dist_back_v6 = (coords_active - depot_coord_v6).norm(p=2, dim=-1) * self.AREA_SIZE
            time_back_v6 = dist_back_v6 / self.VEHICLE_SPEED
            predicted_finish = arrival_time + self.SERVICE_TIME + time_back_v6
            if predicted_finish.dim() == 2:
                predicted_finish = predicted_finish.unsqueeze(1)
            ops_end_mask = (predicted_finish > self.OPERATION_END + 1e-5)
            ops_end_mask[:, :, 0] = False
            mask[:, :, :2 * n_orders + 1] = mask[:, :, :2 * n_orders + 1] | ops_end_mask.to(torch.uint8)

        is_carrying = (self.used_capacity_passenger > 1e-5) | (self.used_capacity_cargo > 1e-5)
        mask[:, :, 0] = mask[:, :, 0] | is_carrying.to(torch.uint8)

        n_visited = self.visited_[:, :, 1:2 * n_orders + 1].sum(-1)
        at_depot_no_work = (self.prev_a == 0) & (n_visited == 0)
        mask[:, :, 0] = mask[:, :, 0] | at_depot_no_work.to(torch.uint8)

        all_done = ((self.visited_[:, :, 1:2 * n_orders + 1].sum(-1) == 2 * n_orders) | (self.rejected_.sum(-1) == n_orders)).to(torch.uint8)
        if Config.HARD_VEHICLE_LIMIT:
            import math as _math
            k_max = max(1, _math.ceil(n_orders * Config.DEFAULT_NUM_VEHICLE_RATIO))
            at_capacity = (self.used_vehicles >= k_max).to(torch.uint8)
            at_depot = (self.prev_a == 0).to(torch.uint8)
            block_all_when_exhausted = at_capacity & at_depot & (1 - all_done)
            if block_all_when_exhausted.any():
                mask[:, :, :2 * n_orders + 1] = mask[:, :, :2 * n_orders + 1] | block_all_when_exhausted[:, :, None].expand_as(mask[:, :, :2 * n_orders + 1])
                mask[:, :, 0] = mask[:, :, 0] | block_all_when_exhausted

        all_order_nodes_masked = mask[:, :, 1:2 * n_orders + 1].bool().all(-1)
        if Config.HARD_VEHICLE_LIMIT:
            import math as _math
            k_max = max(1, _math.ceil(n_orders * Config.DEFAULT_NUM_VEHICLE_RATIO))
            can_start_new_vehicle = (self.used_vehicles < k_max) & (self.prev_a == 0)
            allow_depot_fallback = all_order_nodes_masked & can_start_new_vehicle
        else:
            allow_depot_fallback = all_order_nodes_masked
        mask[:, :, 0] = mask[:, :, 0] & (~allow_depot_fallback).to(torch.uint8)

        reject_candidates = self._deterministic_reject_order(mask)
        reject_allowed = (reject_candidates >= 0) & (~all_done.squeeze(-1).bool())
        mask[:, :, reject_index] = (~reject_allowed).view(-1, 1).to(torch.uint8)

        return mask.bool()

    def update(self, selected):
        selected = selected[:, None]
        n_orders = self.n_orders
        DEPOT = 0
        reject_index = self.reject_index

        ids_flat, coords_active, node_type_active, time_windows_active, demand_p_active, demand_c_active = self._active_views()
        current_mask = self.get_mask()
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

        new_trip_start = torch.where(is_reject, self.trip_start_time, is_depot * new_time + (1 - is_depot) * self.trip_start_time)

        all_order_nodes_masked = current_mask[:, :, 1:2 * n_orders + 1].all(-1).to(torch.long)
        new_deadlock_count = torch.where(
            (is_depot > 0) & (~is_reject),
            self.deadlock_count + all_order_nodes_masked,
            torch.zeros_like(self.deadlock_count),
        )
        new_terminal = self.terminal_ | (new_deadlock_count >= 2)

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
            terminal_=new_terminal,
            i=self.i + 1,
        )

    def all_finished(self):
        all_visited = ((self.visited_[:, :, 1:2 * self.n_orders + 1].sum(-1) == self.n_orders * 2) | (self.rejected_.sum(-1) == self.n_orders))
        at_depot = self.prev_a == 0
        return ((all_visited & at_depot) | self.terminal_).all()

    def get_finished(self):
        all_visited = ((self.visited_[:, :, 1:2 * self.n_orders + 1].sum(-1) == self.n_orders * 2) | (self.rejected_.sum(-1) == self.n_orders))
        at_depot = self.prev_a == 0
        deadlock_finished = self.deadlock_count >= 2
        return ((all_visited & at_depot) | deadlock_finished | self.terminal_).squeeze(-1)

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
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                deadlock_count=self.deadlock_count[key],
                terminal_=self.terminal_[key],
            )
        raise TypeError(f"Invalid key type: {type(key)}")
