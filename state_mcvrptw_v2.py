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

import time

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
    min_orders_per_dispatch: int
    enable_delivery_viability: bool
    enable_viability_fallback: bool
    relax_pickup_commitment_trip_time: bool
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    deadlock_count: torch.Tensor
    deadlock_limit: torch.Tensor
    served_orders_since_dispatch: torch.Tensor
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

    @staticmethod
    def _memo_bucket(memo, name):
        if memo is None:
            return None
        bucket = memo.get(name)
        if bucket is None:
            bucket = {}
            memo[name] = bucket
        return bucket

    @staticmethod
    def _record_benchmark_timing(benchmark_stats, name, seconds, calls=1):
        if benchmark_stats is None:
            return
        seconds_bucket = benchmark_stats.get('seconds')
        calls_bucket = benchmark_stats.get('calls')
        if seconds_bucket is None or calls_bucket is None:
            return
        seconds_bucket[name] = seconds_bucket.get(name, 0.0) + float(seconds)
        calls_bucket[name] = calls_bucket.get(name, 0) + int(calls)

    @staticmethod
    def _record_benchmark_count(benchmark_stats, name, value=1):
        if benchmark_stats is None:
            return
        calls_bucket = benchmark_stats.get('calls')
        if calls_bucket is None:
            return
        calls_bucket[name] = calls_bucket.get(name, 0) + int(value)

    @staticmethod
    def _record_benchmark_max(benchmark_stats, name, value):
        if benchmark_stats is None:
            return
        calls_bucket = benchmark_stats.get('calls')
        if calls_bucket is None:
            return
        calls_bucket[name] = max(calls_bucket.get(name, 0), int(value))

    @staticmethod
    def _open_mask_to_bits(open_mask):
        open_bits = 0
        if open_mask is None:
            return open_bits
        for order_idx in torch.nonzero(open_mask, as_tuple=False).squeeze(-1).tolist():
            open_bits |= 1 << int(order_idx)
        return open_bits

    @staticmethod
    def _iter_open_bits(open_bits):
        remaining = int(open_bits)
        while remaining:
            lowest_bit = remaining & -remaining
            yield lowest_bit.bit_length() - 1
            remaining ^= lowest_bit

    @staticmethod
    def _count_open_bits(open_bits):
        remaining = int(open_bits)
        count = 0
        while remaining:
            remaining &= remaining - 1
            count += 1
        return count

    @staticmethod
    def _stable_time_key(value):
        value = float(value)
        return int(round(value * 1000000.0))

    def _trip_elapsed_key(self, start_time, trip_start_time):
        return self._stable_time_key(float(start_time) - float(trip_start_time))

    def _ride_elapsed_key(self, start_time, pickup_time):
        pickup_time = float(pickup_time)
        if pickup_time < 0:
            return -1
        return self._stable_time_key(float(start_time) - pickup_time)

    def _pickup_node_key(self, order_idx):
        return int(order_idx) + 1

    def _delivery_node_key(self, order_idx):
        return int(order_idx) + self.n_orders + 1

    def _current_start_node_key(self, prev_node):
        prev_node = int(prev_node)
        return None if prev_node == self.reject_index else prev_node

    def _completion_search_cache_key(self, start_coord, start_time, trip_start_time, open_bits,
                                     passenger_orders, passenger_pickup_times, ignore_trip_time=False,
                                     start_node=None, scalar_cache=None):
        if scalar_cache is not None:
            passenger_flags = scalar_cache['passenger_orders']
            pickup_times = scalar_cache['pickup_times']
            passenger_pickup_key = tuple(
                self._ride_elapsed_key(start_time, pickup_times[order_idx]) if passenger_flags[order_idx] else None
                for order_idx in self._iter_open_bits(open_bits)
            )
        else:
            passenger_pickup_key = tuple(
                self._ride_elapsed_key(start_time, passenger_pickup_times[order_idx].item()) if bool(passenger_orders[order_idx].item()) else None
                for order_idx in self._iter_open_bits(open_bits)
            )
        start_key = start_node if start_node is not None else tuple(self._stable_time_key(v) for v in start_coord.tolist())
        return (
            start_key,
            self._stable_time_key(start_time),
            self._trip_elapsed_key(start_time, trip_start_time),
            int(open_bits),
            passenger_pickup_key,
            bool(ignore_trip_time),
        )

    def _step_feasibility_cache_key(self, start_coord, start_time, trip_start_time, order_idx,
                                    passenger_orders, passenger_pickup_times, ignore_trip_time=False,
                                    start_node=None, scalar_cache=None):
        start_key = start_node if start_node is not None else tuple(self._stable_time_key(v) for v in start_coord.tolist())
        if scalar_cache is not None:
            passenger_flags = scalar_cache['passenger_orders']
            pickup_times = scalar_cache['pickup_times']
            ride_key = self._ride_elapsed_key(start_time, pickup_times[order_idx]) if passenger_flags[order_idx] else None
        else:
            ride_key = self._ride_elapsed_key(start_time, passenger_pickup_times[order_idx].item()) if bool(passenger_orders[order_idx].item()) else None
        return (
            start_key,
            self._stable_time_key(start_time),
            self._trip_elapsed_key(start_time, trip_start_time),
            int(order_idx),
            ride_key,
            bool(ignore_trip_time),
        )

    def _build_completion_scalar_caches(self, pickup_coords, delivery_coords, passenger_orders,
                                        passenger_pickup_times, direct_ride_time,
                                        delivery_earliest, delivery_to_depot_time):
        batch_size = pickup_coords.size(0)
        passenger_flags_batch = [[bool(v) for v in row] for row in passenger_orders.tolist()]
        pickup_times_batch = passenger_pickup_times.tolist()
        direct_ride_limits_batch = direct_ride_time.tolist()
        delivery_earliest_batch = delivery_earliest.tolist()
        delivery_to_depot_batch = delivery_to_depot_time.tolist()

        latest_delivery_arrival_batch = [[None] * self.n_orders for _ in range(batch_size)]
        if Config.HARD_PASSENGER_MAX_RIDE_TIME:
            total_limit = Config.PASSENGER_MAX_RIDE_TIME_MINUTES / 60.0
            excess_limit = Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES / 60.0
            for batch_idx in range(batch_size):
                passenger_flags = passenger_flags_batch[batch_idx]
                pickup_times = pickup_times_batch[batch_idx]
                direct_ride_limits = direct_ride_limits_batch[batch_idx]
                latest_delivery_arrival = latest_delivery_arrival_batch[batch_idx]
                for order_idx in range(self.n_orders):
                    if not passenger_flags[order_idx]:
                        continue
                    pickup_time = pickup_times[order_idx]
                    if pickup_time < 0:
                        latest_delivery_arrival[order_idx] = -1.0
                        continue
                    latest_delivery_arrival[order_idx] = min(
                        pickup_time + total_limit,
                        pickup_time + direct_ride_limits[order_idx] + excess_limit,
                    )

        travel_time_scale = self.AREA_SIZE / self.VEHICLE_SPEED
        pickup_to_delivery_batch = (torch.cdist(pickup_coords, delivery_coords, p=2) * travel_time_scale).tolist()
        delivery_to_delivery_batch = (torch.cdist(delivery_coords, delivery_coords, p=2) * travel_time_scale).tolist()

        caches = []
        for batch_idx in range(batch_size):
            travel_time_to_delivery_by_start_node = [None] * (2 * self.n_orders + 1)
            travel_time_to_delivery_by_start_node[0] = delivery_to_depot_batch[batch_idx]
            pickup_to_delivery_time = pickup_to_delivery_batch[batch_idx]
            delivery_to_delivery_time = delivery_to_delivery_batch[batch_idx]
            for order_idx in range(self.n_orders):
                travel_time_to_delivery_by_start_node[self._pickup_node_key(order_idx)] = pickup_to_delivery_time[order_idx]
                travel_time_to_delivery_by_start_node[self._delivery_node_key(order_idx)] = delivery_to_delivery_time[order_idx]
            caches.append({
                'passenger_orders': passenger_flags_batch[batch_idx],
                'pickup_times': pickup_times_batch[batch_idx],
                'direct_ride_time': direct_ride_limits_batch[batch_idx],
                'delivery_earliest': delivery_earliest_batch[batch_idx],
                'delivery_to_depot_time': delivery_to_depot_batch[batch_idx],
                'latest_delivery_arrival': latest_delivery_arrival_batch[batch_idx],
                'travel_time_to_delivery_by_start_node': travel_time_to_delivery_by_start_node,
            })
        return caches

    def _build_completion_scalar_cache(self, pickup_coords, delivery_coords, passenger_orders,
                                       passenger_pickup_times, direct_ride_time,
                                       delivery_earliest, delivery_to_depot_time):
        return self._build_completion_scalar_caches(
            pickup_coords.unsqueeze(0),
            delivery_coords.unsqueeze(0),
            passenger_orders.unsqueeze(0),
            passenger_pickup_times.unsqueeze(0),
            direct_ride_time.unsqueeze(0),
            delivery_earliest.unsqueeze(0),
            delivery_to_depot_time.unsqueeze(0),
        )[0]

    def _completion_step_feasible_from_cache(self, start_time, trip_start_time, order_idx,
                                             scalar_cache, start_node, ignore_trip_time=False):
        delivery_earliest_value = scalar_cache['delivery_earliest'][order_idx]
        delivery_to_depot_value = scalar_cache['delivery_to_depot_time'][order_idx]
        travel_lookup = scalar_cache['travel_time_to_delivery_by_start_node']
        travel_time = travel_lookup[start_node][order_idx]
        arrival_time = float(start_time) + travel_time

        if scalar_cache['passenger_orders'][order_idx]:
            pickup_time = scalar_cache['pickup_times'][order_idx]
            if pickup_time < 0:
                return False, None
            latest_arrival = scalar_cache['latest_delivery_arrival'][order_idx]
            if latest_arrival is not None and arrival_time > latest_arrival + 1e-5:
                return False, None

        service_start = max(arrival_time, delivery_earliest_value)
        finish_time = service_start + self.SERVICE_TIME
        finish_with_return = finish_time + delivery_to_depot_value
        if (not ignore_trip_time) and Config.HARD_MAX_TRIP_TIME and (
            finish_with_return - float(trip_start_time) > self.MAX_TRIP_TIME + 1e-5
        ):
            return False, None
        if Config.HARD_OPERATION_END and (finish_with_return > self.OPERATION_END + 1e-5):
            return False, None
        return True, finish_time

    def _has_feasible_open_completion_bool_dp(self, start_coord, start_time, trip_start_time, open_mask,
                                              delivery_coords, delivery_earliest, delivery_to_depot_time,
                                              passenger_orders, passenger_pickup_times, direct_ride_time,
                                              ignore_trip_time=False, memo=None, open_bits=None,
                                              start_node=None, benchmark_stats=None, scalar_cache=None):
        helper_start = time.perf_counter() if benchmark_stats is not None else None

        def _finalize(result):
            if helper_start is not None:
                self._record_benchmark_timing(
                    benchmark_stats,
                    'pc_completion_bool_dp',
                    time.perf_counter() - helper_start,
                )
            return result

        if open_bits is None:
            open_bits = self._open_mask_to_bits(open_mask)
        if open_bits == 0:
            return _finalize(True)

        order_indices = tuple(self._iter_open_bits(open_bits))
        open_count = len(order_indices)
        if benchmark_stats is not None:
            self._record_benchmark_count(benchmark_stats, 'pc_completion_dp_calls')
            self._record_benchmark_count(benchmark_stats, 'pc_completion_dp_open_bits_sum', open_count)

        full_subset = (1 << open_count) - 1
        inf = float('inf')
        best_finish = [[inf] * open_count for _ in range(1 << open_count)]
        transition_attempts = 0

        if scalar_cache is not None and start_node is not None:
            for pos, order_idx in enumerate(order_indices):
                transition_attempts += 1
                feasible_step, finish_time = self._completion_step_feasible_from_cache(
                    start_time,
                    trip_start_time,
                    order_idx,
                    scalar_cache,
                    start_node,
                    ignore_trip_time=ignore_trip_time,
                )
                if feasible_step:
                    best_finish[1 << pos][pos] = float(finish_time)

            for subset in range(1, full_subset + 1):
                subset_states = best_finish[subset]
                for last_pos, current_finish in enumerate(subset_states):
                    if current_finish == inf:
                        continue
                    if subset == full_subset:
                        if benchmark_stats is not None:
                            self._record_benchmark_count(benchmark_stats, 'pc_completion_dp_success')
                            self._record_benchmark_count(benchmark_stats, 'pc_completion_dp_transition_attempts', transition_attempts)
                        return _finalize(True)

                    last_order_idx = order_indices[last_pos]
                    last_start_node = self._delivery_node_key(last_order_idx)
                    remaining = full_subset ^ subset
                    while remaining:
                        lowest_bit = remaining & -remaining
                        next_pos = lowest_bit.bit_length() - 1
                        remaining ^= lowest_bit
                        next_order_idx = order_indices[next_pos]
                        transition_attempts += 1
                        feasible_step, next_finish = self._completion_step_feasible_from_cache(
                            current_finish,
                            trip_start_time,
                            next_order_idx,
                            scalar_cache,
                            last_start_node,
                            ignore_trip_time=ignore_trip_time,
                        )
                        if not feasible_step:
                            continue
                        next_subset = subset | (1 << next_pos)
                        next_finish = float(next_finish)
                        if next_finish < best_finish[next_subset][next_pos]:
                            best_finish[next_subset][next_pos] = next_finish

            if benchmark_stats is not None:
                self._record_benchmark_count(benchmark_stats, 'pc_completion_dp_failure')
                self._record_benchmark_count(benchmark_stats, 'pc_completion_dp_transition_attempts', transition_attempts)
            return _finalize(False)

        for pos, order_idx in enumerate(order_indices):
            transition_attempts += 1
            feasible_step, _, finish_time = self._delivery_step_feasible(
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
                ignore_trip_time=ignore_trip_time,
                memo=memo,
                start_node=start_node,
                benchmark_stats=benchmark_stats,
                scalar_cache=scalar_cache,
            )
            if feasible_step:
                best_finish[1 << pos][pos] = float(finish_time)

        for subset in range(1, full_subset + 1):
            subset_states = best_finish[subset]
            for last_pos, current_finish in enumerate(subset_states):
                if current_finish == inf:
                    continue
                if subset == full_subset:
                    if benchmark_stats is not None:
                        self._record_benchmark_count(benchmark_stats, 'pc_completion_dp_success')
                        self._record_benchmark_count(benchmark_stats, 'pc_completion_dp_transition_attempts', transition_attempts)
                    return _finalize(True)

                last_order_idx = order_indices[last_pos]
                last_coord = delivery_coords[last_order_idx]
                last_start_node = self._delivery_node_key(last_order_idx)
                remaining = full_subset ^ subset
                while remaining:
                    lowest_bit = remaining & -remaining
                    next_pos = lowest_bit.bit_length() - 1
                    remaining ^= lowest_bit
                    next_order_idx = order_indices[next_pos]
                    transition_attempts += 1
                    feasible_step, _, next_finish = self._delivery_step_feasible(
                        last_coord,
                        current_finish,
                        trip_start_time,
                        next_order_idx,
                        delivery_coords,
                        delivery_earliest,
                        delivery_to_depot_time,
                        passenger_orders,
                        passenger_pickup_times,
                        direct_ride_time,
                        ignore_trip_time=ignore_trip_time,
                        memo=memo,
                        start_node=last_start_node,
                        benchmark_stats=benchmark_stats,
                        scalar_cache=scalar_cache,
                    )
                    if not feasible_step:
                        continue
                    next_subset = subset | (1 << next_pos)
                    next_finish = float(next_finish)
                    if next_finish < best_finish[next_subset][next_pos]:
                        best_finish[next_subset][next_pos] = next_finish

        if benchmark_stats is not None:
            self._record_benchmark_count(benchmark_stats, 'pc_completion_dp_failure')
            self._record_benchmark_count(benchmark_stats, 'pc_completion_dp_transition_attempts', transition_attempts)
        return _finalize(False)

    def _has_feasible_open_completion(self, start_coord, start_time, trip_start_time, open_mask,
                                      delivery_coords, delivery_earliest, delivery_to_depot_time,
                                      passenger_orders, passenger_pickup_times, direct_ride_time,
                                      return_reason=False, ignore_trip_time=False, memo=None,
                                      open_bits=None, start_node=None, benchmark_stats=None, recursion_depth=0,
                                      scalar_cache=None):
        helper_start = time.perf_counter() if benchmark_stats is not None else None
        if benchmark_stats is not None:
            self._record_benchmark_count(benchmark_stats, 'pc_has_feasible_open_completion_calls')
            self._record_benchmark_count(benchmark_stats, 'pc_completion_recursion_depth_sum', recursion_depth)
            self._record_benchmark_max(benchmark_stats, 'pc_completion_max_recursion_depth', recursion_depth)

        def _finalize(result):
            if helper_start is not None:
                self._record_benchmark_timing(
                    benchmark_stats,
                    'pc_has_feasible_open_completion',
                    time.perf_counter() - helper_start,
                )
            return result

        if open_bits is None:
            open_bits = self._open_mask_to_bits(open_mask)
        if benchmark_stats is not None:
            self._record_benchmark_count(benchmark_stats, 'pc_completion_open_bits_sum', self._count_open_bits(open_bits))
        if open_bits == 0:
            if benchmark_stats is not None:
                self._record_benchmark_count(benchmark_stats, 'pc_completion_terminal_success')
            return _finalize((True, None) if return_reason else True)
        if self._count_open_bits(open_bits) > 6:
            if benchmark_stats is not None:
                self._record_benchmark_count(benchmark_stats, 'pc_completion_pruned_open_over_6')
            return _finalize((False, 'open_over_6') if return_reason else False)

        completion_memo_name = 'completion' if return_reason else 'completion_bool'
        completion_memo = self._memo_bucket(memo, completion_memo_name)
        exact_completion_memo = None if return_reason or memo is None else memo.get('completion')
        cache_key = None
        if completion_memo is not None:
            cache_key = self._completion_search_cache_key(
                start_coord,
                start_time,
                trip_start_time,
                open_bits,
                passenger_orders,
                passenger_pickup_times,
                ignore_trip_time=ignore_trip_time,
                start_node=start_node,
                scalar_cache=scalar_cache,
            )
            self._record_benchmark_count(benchmark_stats, 'pc_completion_memo_lookups')
            cached = completion_memo.get(cache_key)
            if cached is None and exact_completion_memo is not None:
                cached = exact_completion_memo.get(cache_key)
            if cached is not None:
                self._record_benchmark_count(benchmark_stats, 'pc_completion_memo_hits')
                return _finalize(cached if return_reason else cached[0])
            self._record_benchmark_count(benchmark_stats, 'pc_completion_memo_misses')

        if (not return_reason) and (not ignore_trip_time):
            dp_result = self._has_feasible_open_completion_bool_dp(
                start_coord,
                start_time,
                trip_start_time,
                None,
                delivery_coords,
                delivery_earliest,
                delivery_to_depot_time,
                passenger_orders,
                passenger_pickup_times,
                direct_ride_time,
                ignore_trip_time=False,
                memo=memo,
                open_bits=open_bits,
                start_node=start_node,
                benchmark_stats=benchmark_stats,
                scalar_cache=scalar_cache,
            )
            if completion_memo is not None and cache_key is not None:
                completion_memo[cache_key] = (dp_result, None)
                self._record_benchmark_count(benchmark_stats, 'pc_completion_memo_stores')
            if benchmark_stats is not None:
                self._record_benchmark_count(benchmark_stats, 'pc_completion_dp_path_used')
            return _finalize(dp_result)

        if open_bits & (open_bits - 1) == 0:
            if benchmark_stats is not None:
                self._record_benchmark_count(benchmark_stats, 'pc_completion_single_open_cases')
            order_idx = open_bits.bit_length() - 1
            if return_reason:
                step_result = self._delivery_step_feasible(
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
                    return_reason=True,
                    ignore_trip_time=ignore_trip_time,
                    memo=memo,
                    start_node=start_node,
                    benchmark_stats=benchmark_stats,
                    scalar_cache=scalar_cache,
                )
                feasible, _, _, reason = step_result
                final_result = (feasible, reason)
            else:
                feasible, _, _ = self._delivery_step_feasible(
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
                    ignore_trip_time=ignore_trip_time,
                    memo=memo,
                    start_node=start_node,
                    benchmark_stats=benchmark_stats,
                    scalar_cache=scalar_cache,
                )
                final_result = (feasible, None)
            if completion_memo is not None and cache_key is not None:
                completion_memo[cache_key] = final_result
                self._record_benchmark_count(benchmark_stats, 'pc_completion_memo_stores')
            if benchmark_stats is not None:
                self._record_benchmark_count(
                    benchmark_stats,
                    'pc_completion_terminal_success' if final_result[0] else 'pc_completion_terminal_failure',
                )
            return _finalize(final_result if return_reason else final_result[0])

        branch_attempts = self._count_open_bits(open_bits)
        if not return_reason:
            feasible_steps = self._enumerate_feasible_delivery_steps(
                start_coord,
                start_time,
                trip_start_time,
                None,
                delivery_coords,
                delivery_earliest,
                delivery_to_depot_time,
                passenger_orders,
                passenger_pickup_times,
                direct_ride_time,
                memo=memo,
                open_bits=open_bits,
                start_node=start_node,
                benchmark_stats=benchmark_stats,
                scalar_cache=scalar_cache,
                ignore_trip_time=ignore_trip_time,
                collect_failure_reasons=False,
            )
            for order_idx, finish_time in feasible_steps:
                child_open_bits = open_bits & ~(1 << order_idx)
                child_start_node = self._delivery_node_key(order_idx)
                next_coord = delivery_coords[order_idx]
                if self._has_feasible_open_completion(
                    next_coord,
                    finish_time,
                    trip_start_time,
                    None,
                    delivery_coords,
                    delivery_earliest,
                    delivery_to_depot_time,
                    passenger_orders,
                    passenger_pickup_times,
                    direct_ride_time,
                    ignore_trip_time=ignore_trip_time,
                    memo=memo,
                    open_bits=child_open_bits,
                    start_node=child_start_node,
                    benchmark_stats=benchmark_stats,
                    recursion_depth=recursion_depth + 1,
                    scalar_cache=scalar_cache,
                ):
                    final_result = (True, None)
                    if completion_memo is not None and cache_key is not None:
                        completion_memo[cache_key] = final_result
                        self._record_benchmark_count(benchmark_stats, 'pc_completion_memo_stores')
                    if benchmark_stats is not None:
                        self._record_benchmark_count(benchmark_stats, 'pc_completion_success_paths')
                        self._record_benchmark_count(benchmark_stats, 'pc_completion_branch_attempts', branch_attempts)
                    return _finalize(True)
            final_result = (False, None)
            if completion_memo is not None and cache_key is not None:
                completion_memo[cache_key] = final_result
                self._record_benchmark_count(benchmark_stats, 'pc_completion_memo_stores')
            if benchmark_stats is not None:
                self._record_benchmark_count(benchmark_stats, 'pc_completion_failure_paths')
                self._record_benchmark_count(benchmark_stats, 'pc_completion_branch_attempts', branch_attempts)
            return _finalize(False)

        feasible_steps, failure_reason_list = self._enumerate_feasible_delivery_steps(
            start_coord,
            start_time,
            trip_start_time,
            None,
            delivery_coords,
            delivery_earliest,
            delivery_to_depot_time,
            passenger_orders,
            passenger_pickup_times,
            direct_ride_time,
            memo=memo,
            open_bits=open_bits,
            start_node=start_node,
            benchmark_stats=benchmark_stats,
            scalar_cache=scalar_cache,
            ignore_trip_time=ignore_trip_time,
            collect_failure_reasons=True,
        )
        failure_reasons = set(failure_reason_list)
        for order_idx, finish_time in feasible_steps:
            child_open_bits = open_bits & ~(1 << order_idx)
            child_start_node = self._delivery_node_key(order_idx)
            next_coord = delivery_coords[order_idx]

            recursive_result = self._has_feasible_open_completion(
                next_coord,
                finish_time,
                trip_start_time,
                None,
                delivery_coords,
                delivery_earliest,
                delivery_to_depot_time,
                passenger_orders,
                passenger_pickup_times,
                direct_ride_time,
                return_reason=True,
                ignore_trip_time=ignore_trip_time,
                memo=memo,
                open_bits=child_open_bits,
                start_node=child_start_node,
                benchmark_stats=benchmark_stats,
                recursion_depth=recursion_depth + 1,
                scalar_cache=scalar_cache,
            )
            recursive_feasible, recursive_reason = recursive_result
            if recursive_feasible:
                final_result = (True, None)
                if completion_memo is not None and cache_key is not None:
                    completion_memo[cache_key] = final_result
                    self._record_benchmark_count(benchmark_stats, 'pc_completion_memo_stores')
                if benchmark_stats is not None:
                    self._record_benchmark_count(benchmark_stats, 'pc_completion_success_paths')
                    self._record_benchmark_count(benchmark_stats, 'pc_completion_branch_attempts', branch_attempts)
                return _finalize(final_result)
            if recursive_reason is not None:
                failure_reasons.add(recursive_reason)

        if not failure_reasons:
            final_result = (False, 'unknown')
        elif len(failure_reasons) == 1:
            final_result = (False, next(iter(failure_reasons)))
        else:
            final_result = (False, 'mixed')
        if completion_memo is not None and cache_key is not None:
            completion_memo[cache_key] = final_result
            self._record_benchmark_count(benchmark_stats, 'pc_completion_memo_stores')
        if benchmark_stats is not None:
            self._record_benchmark_count(benchmark_stats, 'pc_completion_failure_paths')
            self._record_benchmark_count(benchmark_stats, 'pc_completion_branch_attempts', branch_attempts)
        return _finalize(final_result)

    def _delivery_step_feasible(self, start_coord, start_time, trip_start_time, order_idx,
                                delivery_coords, delivery_earliest, delivery_to_depot_time,
                                passenger_orders, passenger_pickup_times, direct_ride_time,
                                return_reason=False, ignore_trip_time=False, memo=None,
                                start_node=None, benchmark_stats=None, scalar_cache=None):
        helper_start = time.perf_counter() if benchmark_stats is not None else None
        if benchmark_stats is not None:
            self._record_benchmark_count(benchmark_stats, 'pc_delivery_step_feasible_calls')

        def _finalize(result):
            if helper_start is not None:
                self._record_benchmark_timing(
                    benchmark_stats,
                    'pc_delivery_step_feasible',
                    time.perf_counter() - helper_start,
                )
            return result

        def _return(result):
            if step_memo is not None and cache_key is not None:
                step_memo[cache_key] = result
                self._record_benchmark_count(benchmark_stats, 'pc_step_memo_stores')
            self._record_benchmark_count(benchmark_stats, 'pc_step_reason_' + ('ok' if result[0] else str(result[3] or 'unknown')))
            return _finalize(result if return_reason else result[:3])

        step_memo = self._memo_bucket(memo, 'delivery_step')
        cache_key = None
        if step_memo is not None:
            self._record_benchmark_count(benchmark_stats, 'pc_step_memo_lookups')
            cache_key = self._step_feasibility_cache_key(
                start_coord,
                start_time,
                trip_start_time,
                order_idx,
                passenger_orders,
                passenger_pickup_times,
                ignore_trip_time=ignore_trip_time,
                start_node=start_node,
                scalar_cache=scalar_cache,
            )
            cached = step_memo.get(cache_key)
            if cached is not None:
                self._record_benchmark_count(benchmark_stats, 'pc_step_memo_hits')
                self._record_benchmark_count(benchmark_stats, 'pc_step_reason_' + ('ok' if cached[0] else str(cached[3] or 'unknown')))
                return _finalize(cached if return_reason else cached[:3])
            self._record_benchmark_count(benchmark_stats, 'pc_step_memo_misses')

        delivery_coord = delivery_coords[order_idx]
        current_time = float(start_time)
        trip_start = float(trip_start_time)

        if scalar_cache is not None:
            passenger_flag = scalar_cache['passenger_orders'][order_idx]
            delivery_earliest_value = scalar_cache['delivery_earliest'][order_idx]
            delivery_to_depot_value = scalar_cache['delivery_to_depot_time'][order_idx]
            travel_lookup = scalar_cache.get('travel_time_to_delivery_by_start_node')
            if start_node is not None and travel_lookup is not None and travel_lookup[start_node] is not None:
                travel_time = travel_lookup[start_node][order_idx]
            else:
                travel_time = float(((delivery_coord - start_coord).norm(p=2) * self.AREA_SIZE / self.VEHICLE_SPEED).item())
            arrival_time = current_time + travel_time
            if passenger_flag:
                pickup_time = scalar_cache['pickup_times'][order_idx]
                if pickup_time < 0:
                    return _return((False, None, None, 'missing_pickup_time'))
                latest_arrival = scalar_cache['latest_delivery_arrival'][order_idx]
                if latest_arrival is not None and arrival_time > latest_arrival + 1e-5:
                    return _return((False, None, None, 'ride_time'))
            else:
                pickup_time = None
        else:
            travel_time = float(((delivery_coord - start_coord).norm(p=2) * self.AREA_SIZE / self.VEHICLE_SPEED).item())
            arrival_time = current_time + travel_time
            passenger_flag = bool(passenger_orders[order_idx].item())
            pickup_time = float(passenger_pickup_times[order_idx].item()) if passenger_flag else None
            delivery_earliest_value = float(delivery_earliest[order_idx].item())
            delivery_to_depot_value = float(delivery_to_depot_time[order_idx].item())
            if passenger_flag:
                direct_ride_limit = float(direct_ride_time[order_idx].item())
                if pickup_time < 0:
                    return _return((False, None, None, 'missing_pickup_time'))
                ride_time = arrival_time - pickup_time
                excess_ride_time = max(ride_time - direct_ride_limit, 0.0)
                if Config.HARD_PASSENGER_MAX_RIDE_TIME and (
                    ride_time > Config.PASSENGER_MAX_RIDE_TIME_MINUTES / 60.0 + 1e-5
                    or excess_ride_time > Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES / 60.0 + 1e-5
                ):
                    return _return((False, None, None, 'ride_time'))

        service_start = max(arrival_time, delivery_earliest_value)
        finish_time = service_start + self.SERVICE_TIME
        finish_with_return = finish_time + delivery_to_depot_value
        if (not ignore_trip_time) and Config.HARD_MAX_TRIP_TIME and (finish_with_return - trip_start > self.MAX_TRIP_TIME + 1e-5):
            return _return((False, None, None, 'trip_time'))
        if Config.HARD_OPERATION_END and (finish_with_return > self.OPERATION_END + 1e-5):
            return _return((False, None, None, 'ops_end'))
        return _return((True, delivery_coord, finish_time, None))

    def _has_any_physical_delivery_step(self, start_coord, start_time, trip_start_time, open_mask,
                                        delivery_coords, delivery_earliest, delivery_to_depot_time,
                                        passenger_orders, passenger_pickup_times, direct_ride_time,
                                        memo=None, open_bits=None, start_node=None, benchmark_stats=None,
                                        scalar_cache=None):
        helper_start = time.perf_counter() if benchmark_stats is not None else None

        def _finalize(result):
            if helper_start is not None:
                self._record_benchmark_timing(
                    benchmark_stats,
                    'pc_has_any_physical_delivery_step',
                    time.perf_counter() - helper_start,
                )
            return result

        if open_bits is None:
            open_bits = self._open_mask_to_bits(open_mask)
        if open_bits == 0:
            return _finalize(True)
        for order_idx in self._iter_open_bits(open_bits):
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
                memo=memo,
                start_node=start_node,
                benchmark_stats=benchmark_stats,
                scalar_cache=scalar_cache,
            )
            if feasible_step:
                return _finalize(True)
        return _finalize(False)

    def _delivery_search_cache_key(self, start_coord, start_time, trip_start_time, open_bits,
                                   passenger_orders, passenger_pickup_times, allow_fallback=False,
                                   start_node=None, scalar_cache=None):
        if scalar_cache is not None:
            passenger_flags = scalar_cache['passenger_orders']
            pickup_times = scalar_cache['pickup_times']
            passenger_pickup_key = tuple(
                self._ride_elapsed_key(start_time, pickup_times[order_idx]) if passenger_flags[order_idx] else None
                for order_idx in self._iter_open_bits(open_bits)
            )
        else:
            passenger_pickup_key = tuple(
                self._ride_elapsed_key(start_time, passenger_pickup_times[order_idx].item()) if bool(passenger_orders[order_idx].item()) else None
                for order_idx in self._iter_open_bits(open_bits)
            )
        start_key = start_node if start_node is not None else tuple(self._stable_time_key(v) for v in start_coord.tolist())
        return (
            start_key,
            self._stable_time_key(start_time),
            self._trip_elapsed_key(start_time, trip_start_time),
            int(open_bits),
            passenger_pickup_key,
            bool(allow_fallback),
        )

    def _enumerate_feasible_delivery_steps(self, start_coord, start_time, trip_start_time, open_mask,
                                           delivery_coords, delivery_earliest, delivery_to_depot_time,
                                           passenger_orders, passenger_pickup_times, direct_ride_time,
                                           memo=None, open_bits=None, start_node=None, benchmark_stats=None,
                                           scalar_cache=None, ignore_trip_time=False, collect_failure_reasons=False):
        helper_start = time.perf_counter() if benchmark_stats is not None else None

        def _finalize(result):
            if helper_start is not None:
                self._record_benchmark_timing(
                    benchmark_stats,
                    'pc_enumerate_feasible_delivery_steps',
                    time.perf_counter() - helper_start,
                )
            return result

        if open_bits is None:
            open_bits = self._open_mask_to_bits(open_mask)
        if open_bits == 0:
            empty = (tuple(), tuple()) if collect_failure_reasons else tuple()
            return _finalize(empty)

        enum_memo = self._memo_bucket(memo, 'delivery_step_enum_reason' if collect_failure_reasons else 'delivery_step_enum')
        cache_key = None
        if enum_memo is not None:
            cache_key = self._completion_search_cache_key(
                start_coord,
                start_time,
                trip_start_time,
                open_bits,
                passenger_orders,
                passenger_pickup_times,
                ignore_trip_time=ignore_trip_time,
                start_node=start_node,
                scalar_cache=scalar_cache,
            )
            cached = enum_memo.get(cache_key)
            if cached is not None:
                return _finalize(cached)

        feasible_steps = []
        failure_reasons = set()
        for order_idx in self._iter_open_bits(open_bits):
            if collect_failure_reasons:
                step_feasible, _, finish_time, step_reason = self._delivery_step_feasible(
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
                    return_reason=True,
                    ignore_trip_time=ignore_trip_time,
                    memo=memo,
                    start_node=start_node,
                    benchmark_stats=benchmark_stats,
                    scalar_cache=scalar_cache,
                )
                if step_feasible:
                    feasible_steps.append((int(order_idx), float(finish_time)))
                elif step_reason is not None:
                    failure_reasons.add(step_reason)
            else:
                step_feasible, _, finish_time = self._delivery_step_feasible(
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
                    ignore_trip_time=ignore_trip_time,
                    memo=memo,
                    start_node=start_node,
                    benchmark_stats=benchmark_stats,
                    scalar_cache=scalar_cache,
                )
                if step_feasible:
                    feasible_steps.append((int(order_idx), float(finish_time)))

        feasible_steps.sort(key=lambda item: (item[1], item[0]))
        if collect_failure_reasons:
            result = (tuple(feasible_steps), tuple(sorted(failure_reasons)))
        else:
            result = tuple(feasible_steps)
        if enum_memo is not None and cache_key is not None:
            enum_memo[cache_key] = result
        return _finalize(result)

    def _get_legal_delivery_orders(self, start_coord, start_time, trip_start_time, open_mask,
                                   delivery_coords, delivery_earliest, delivery_to_depot_time,
                                   passenger_orders, passenger_pickup_times, direct_ride_time,
                                   allow_fallback=False, memo=None, open_bits=None, start_node=None,
                                   benchmark_stats=None, scalar_cache=None):
        helper_start = time.perf_counter() if benchmark_stats is not None else None

        def _finalize(result):
            if helper_start is not None:
                self._record_benchmark_timing(
                    benchmark_stats,
                    'pc_get_legal_delivery_orders',
                    time.perf_counter() - helper_start,
                )
            return result

        if open_bits is None:
            open_bits = self._open_mask_to_bits(open_mask)
        if open_bits == 0:
            return _finalize(([], [], False))

        delivery_memo = self._memo_bucket(memo, 'delivery_search')
        cache_key = None
        if delivery_memo is not None:
            cache_key = self._delivery_search_cache_key(
                start_coord,
                start_time,
                trip_start_time,
                open_bits,
                passenger_orders,
                passenger_pickup_times,
                allow_fallback=allow_fallback,
                start_node=start_node,
                scalar_cache=scalar_cache,
            )
            cached = delivery_memo.get(cache_key)
            if cached is not None:
                return _finalize(cached)

        feasible_steps = self._enumerate_feasible_delivery_steps(
            start_coord,
            start_time,
            trip_start_time,
            None,
            delivery_coords,
            delivery_earliest,
            delivery_to_depot_time,
            passenger_orders,
            passenger_pickup_times,
            direct_ride_time,
            memo=memo,
            open_bits=open_bits,
            start_node=start_node,
            benchmark_stats=benchmark_stats,
            scalar_cache=scalar_cache,
            collect_failure_reasons=False,
        )
        physical_orders = [order_idx for order_idx, _finish_time in feasible_steps]
        viable_orders = []
        for order_idx, finish_time in feasible_steps:
            next_coord = delivery_coords[order_idx]
            if self._has_legal_delivery_path(
                next_coord,
                finish_time,
                trip_start_time,
                None,
                delivery_coords,
                delivery_earliest,
                delivery_to_depot_time,
                passenger_orders,
                passenger_pickup_times,
                direct_ride_time,
                memo=memo,
                open_bits=open_bits & ~(1 << order_idx),
                start_node=self._delivery_node_key(order_idx),
                benchmark_stats=benchmark_stats,
                scalar_cache=scalar_cache,
            ):
                viable_orders.append(order_idx)

        used_fallback = bool(allow_fallback and len(viable_orders) == 0 and len(physical_orders) > 0)
        result = (physical_orders if used_fallback else viable_orders, physical_orders, used_fallback)
        if delivery_memo is not None and cache_key is not None:
            delivery_memo[cache_key] = result
        return _finalize(result)

    def _has_legal_delivery_path(self, start_coord, start_time, trip_start_time, open_mask,
                                 delivery_coords, delivery_earliest, delivery_to_depot_time,
                                 passenger_orders, passenger_pickup_times, direct_ride_time,
                                 memo=None, open_bits=None, start_node=None, benchmark_stats=None,
                                 scalar_cache=None):
        helper_start = time.perf_counter() if benchmark_stats is not None else None

        def _finalize(result):
            if helper_start is not None:
                self._record_benchmark_timing(
                    benchmark_stats,
                    'pc_has_legal_delivery_path',
                    time.perf_counter() - helper_start,
                )
            return result

        if open_bits is None:
            open_bits = self._open_mask_to_bits(open_mask)
        if open_bits == 0:
            return _finalize(True)
        legal_orders, _, _ = self._get_legal_delivery_orders(
            start_coord,
            start_time,
            trip_start_time,
            None,
            delivery_coords,
            delivery_earliest,
            delivery_to_depot_time,
            passenger_orders,
            passenger_pickup_times,
            direct_ride_time,
            allow_fallback=False,
            memo=memo,
            open_bits=open_bits,
            start_node=start_node,
            benchmark_stats=benchmark_stats,
            scalar_cache=scalar_cache,
        )
        return _finalize(len(legal_orders) > 0)

    def _evaluate_post_pickup_open_delivery(self, start_coord, start_time, trip_start_time, open_mask,
                                            delivery_coords, delivery_earliest, delivery_to_depot_time,
                                            passenger_orders, passenger_pickup_times, direct_ride_time,
                                            memo=None, open_bits=None, start_node=None, benchmark_stats=None,
                                            scalar_cache=None):
        helper_start = time.perf_counter() if benchmark_stats is not None else None

        def _finalize(result):
            if helper_start is not None:
                self._record_benchmark_timing(
                    benchmark_stats,
                    'pc_evaluate_post_pickup_open_delivery',
                    time.perf_counter() - helper_start,
                )
            return result

        if open_bits is None:
            open_bits = self._open_mask_to_bits(open_mask)
        if open_bits == 0:
            return _finalize((True, True))
        if self.enable_delivery_viability:
            feasible_steps = self._enumerate_feasible_delivery_steps(
                start_coord,
                start_time,
                trip_start_time,
                None,
                delivery_coords,
                delivery_earliest,
                delivery_to_depot_time,
                passenger_orders,
                passenger_pickup_times,
                direct_ride_time,
                memo=memo,
                open_bits=open_bits,
                start_node=start_node,
                benchmark_stats=benchmark_stats,
                scalar_cache=scalar_cache,
                collect_failure_reasons=False,
            )
            has_physical = len(feasible_steps) > 0
            for order_idx, finish_time in feasible_steps:
                next_coord = delivery_coords[order_idx]
                if self._has_legal_delivery_path(
                    next_coord,
                    finish_time,
                    trip_start_time,
                    None,
                    delivery_coords,
                    delivery_earliest,
                    delivery_to_depot_time,
                    passenger_orders,
                    passenger_pickup_times,
                    direct_ride_time,
                    memo=memo,
                    open_bits=open_bits & ~(1 << order_idx),
                    start_node=self._delivery_node_key(order_idx),
                    benchmark_stats=benchmark_stats,
                    scalar_cache=scalar_cache,
                ):
                    return _finalize((True, True))
            if self.enable_viability_fallback:
                return _finalize((has_physical, has_physical))
            return _finalize((False, has_physical))

        has_physical = self._has_any_physical_delivery_step(
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
            memo=memo,
            open_bits=open_bits,
            start_node=start_node,
            benchmark_stats=benchmark_stats,
            scalar_cache=scalar_cache,
        )
        return _finalize((has_physical, has_physical))

    def _has_post_pickup_next_delivery(self, start_coord, start_time, trip_start_time, open_mask,
                                       delivery_coords, delivery_earliest, delivery_to_depot_time,
                                       passenger_orders, passenger_pickup_times, direct_ride_time,
                                       open_bits=None, start_node=None):
        has_feasible, _ = self._evaluate_post_pickup_open_delivery(
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
            open_bits=open_bits,
            start_node=start_node,
        )
        return has_feasible

    def _has_feasible_open_delivery_from_state(self):
        n_orders = self.n_orders
        if n_orders == 0:
            return False, False

        open_mask = self.get_open_started_mask()[0]
        if not bool(open_mask.any().item()):
            return False, False

        ids_flat, coords_active, node_type_active, time_windows_active, _, _ = self._active_views()
        del ids_flat
        coords_batch = coords_active[0]
        node_type_batch = node_type_active[0]
        time_windows_batch = time_windows_active[0]

        start_coord = self.cur_coord[0, 0]
        start_time = self.current_time[0, 0]
        trip_start_time = self.current_time[0, 0] if int(self.prev_a[0, 0].item()) == 0 else self.trip_start_time[0, 0]
        delivery_coords = coords_batch[n_orders + 1:2 * n_orders + 1]
        delivery_earliest = time_windows_batch[n_orders + 1:2 * n_orders + 1, 0]
        delivery_to_depot_time = (
            (delivery_coords - coords_batch[0:1]).norm(p=2, dim=-1) * self.AREA_SIZE / self.VEHICLE_SPEED
        )
        passenger_orders = (node_type_batch[1:n_orders + 1] == 1)
        passenger_pickup_times = self.passenger_pickup_time.squeeze(1)[0]
        direct_ride_time = (
            (coords_batch[1:n_orders + 1] - delivery_coords).norm(p=2, dim=-1) * self.AREA_SIZE / self.VEHICLE_SPEED
        )

        return self._evaluate_post_pickup_open_delivery(
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
        )

    @staticmethod
    def initialize(
        input_data,
        visited_dtype=torch.uint8,
        allow_reject=True,
        deadlock_limit=2,
        max_concurrent_open_orders=1,
        min_orders_per_dispatch=4,
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
            min_orders_per_dispatch=max(int(min_orders_per_dispatch), 1),
            enable_delivery_viability=bool(enable_delivery_viability),
            enable_viability_fallback=bool(enable_viability_fallback),
            relax_pickup_commitment_trip_time=bool(relax_pickup_commitment_trip_time),
            lengths=torch.zeros(batch_size, 1, device=device),
            cur_coord=depot[:, None, :],
            deadlock_count=torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            deadlock_limit=torch.full((batch_size, 1), max(int(deadlock_limit), 1), dtype=torch.int64, device=device),
            served_orders_since_dispatch=torch.zeros(batch_size, 1, dtype=torch.int64, device=device),
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

    def get_mask(self, return_debug=False, skip_pickup_commitment=False, benchmark_stats=None):
        batch_size = self.ids.size(0)
        n_orders = self.n_orders
        device = self.coords.device

        def _record(name, start_time):
            if benchmark_stats is None or start_time is None:
                return
            self._record_benchmark_timing(benchmark_stats, name, time.perf_counter() - start_time)

        phase_start = time.perf_counter() if benchmark_stats is not None else None
        ids_flat, coords_active, node_type_active, time_windows_active, demand_p_full, demand_c_full = self._active_views()
        _record('mask_active_views', phase_start)

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
                'diag_depot_min_orders_block',
                'diag_depot_fallback_used',
                'diag_reject_candidate_available',
                'diag_reject_allowed',
                'diag_reject_predeparture_available',
                'diag_reject_inroute_available',
                'diag_reject_dead_end_inroute_available',
            ]
            debug = {key: torch.zeros(batch_size, device=device) for key in debug_keys}

        phase_start = time.perf_counter() if benchmark_stats is not None else None
        service_mask = self.visited[:, :, 1:2 * n_orders + 1].bool().squeeze(1).clone()
        if return_debug:
            debug['diag_mask_visited'] = service_mask.sum(1).float()
        _record('mask_init_visited', phase_start)

        phase_start = time.perf_counter() if benchmark_stats is not None else None
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
        _record('mask_precedence', phase_start)

        phase_start = time.perf_counter() if benchmark_stats is not None else None
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
        _record('mask_capacity', phase_start)

        phase_start = time.perf_counter() if benchmark_stats is not None else None
        dist_to_nodes = (coords_active - self.cur_coord).norm(p=2, dim=-1) * self.AREA_SIZE
        travel_time = dist_to_nodes / self.VEHICLE_SPEED
        arrival_time = self.current_time + travel_time

        tw_end = time_windows_active[:, :, 1]
        node_positions = torch.arange(coords_active.size(1), device=device)[None, :]
        is_passenger_node = node_type_active == 1
        is_cargo_node = node_type_active == 0
        is_pickup_node = (node_positions >= 1) & (node_positions <= n_orders)

        pickup_hard_gate = (
            (Config.HARD_PASSENGER_PICKUP_TIMEWINDOW & is_passenger_node)
            | (Config.HARD_CARGO_PICKUP_TIMEWINDOW & is_cargo_node)
        )
        pickup_hard_mask_full = pickup_hard_gate & is_pickup_node & (arrival_time > tw_end + 1e-5)
        before_mask = service_mask.clone()
        service_mask |= pickup_hard_mask_full[:, 1:2 * n_orders + 1]
        if return_debug:
            debug['diag_mask_pickup_tw'] = (service_mask & ~before_mask).sum(1).float()
        _record('mask_pickup_tw', phase_start)

        phase_start = time.perf_counter() if benchmark_stats is not None else None
        open_started_mask = self.get_open_started_mask()
        open_started_count = self.get_open_started_count().squeeze(1)
        open_started = open_started_count > 0
        if return_debug:
            debug['diag_open_started_count'] = open_started_count.float()
            debug['diag_open_started_eq2'] = (open_started_count == 2).float()
        _record('mask_open_started', phase_start)

        phase_start = time.perf_counter() if benchmark_stats is not None else None
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
        _record('mask_ride_time', phase_start)

        phase_start = time.perf_counter() if benchmark_stats is not None else None
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
        _record('mask_trip_time', phase_start)

        phase_start = time.perf_counter() if benchmark_stats is not None else None
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
        _record('mask_ops_end', phase_start)

        phase_start = time.perf_counter() if benchmark_stats is not None else None
        pickup_commitment_mask = torch.zeros(batch_size, n_orders, dtype=torch.bool, device=device)
        if n_orders > 0:
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
            passenger_pickup_times_before = self.passenger_pickup_time.squeeze(1)
            direct_ride_time = direct_distance / self.VEHICLE_SPEED
            current_time_flat = self.current_time.squeeze(1)
            trip_start_time_flat = self.trip_start_time.squeeze(1)
            prev_a_flat = self.prev_a.squeeze(1)
            cur_coord_flat = self.cur_coord.squeeze(1)
            open_mask_full = open_started_mask
            shared_search_memo = {}
            batch_scalar_caches = self._build_completion_scalar_caches(
                pickup_coords,
                delivery_coords,
                passenger_orders,
                passenger_pickup_times_before,
                direct_ride_time,
                delivery_earliest,
                delivery_to_depot_time,
            )
            requires_post_pickup_delivery_check = self.relax_pickup_commitment_trip_time

        if n_orders > 0 and (not skip_pickup_commitment):
            pickup_physical_feasible = ~service_mask[:, :n_orders]

            for batch_idx in range(batch_size):
                candidate_indices = torch.nonzero(pickup_physical_feasible[batch_idx], as_tuple=False).squeeze(-1)
                if candidate_indices.numel() == 0:
                    continue

                candidate_index_list = candidate_indices.tolist()
                batch_open_count = int(open_started_count[batch_idx].item())
                second_pickup_feasible = 0
                second_pickup_blocked = 0
                if batch_open_count >= self.max_concurrent_open_orders:
                    pickup_commitment_mask[batch_idx, candidate_indices] = True
                    second_pickup_blocked = len(candidate_index_list)
                    if return_debug:
                        debug['diag_pickup_commitment_block_by_k'][batch_idx] += float(len(candidate_index_list))
                else:
                    batch_open_bits = self._open_mask_to_bits(open_before[batch_idx])
                    batch_pickup_coords = pickup_coords[batch_idx]
                    batch_pickup_finish = pickup_finish[batch_idx]
                    batch_trip_start_after_pickup = trip_start_after_pickup[batch_idx]
                    batch_delivery_coords = delivery_coords[batch_idx]
                    batch_delivery_earliest = delivery_earliest[batch_idx]
                    batch_delivery_to_depot_time = delivery_to_depot_time[batch_idx]
                    batch_passenger_orders = passenger_orders[batch_idx]
                    batch_passenger_pickup_times_before = passenger_pickup_times_before[batch_idx]
                    batch_direct_ride_time = direct_ride_time[batch_idx]
                    batch_prev_node = self._current_start_node_key(prev_a_flat[batch_idx].item())
                    batch_scalar_cache = batch_scalar_caches[batch_idx]
                    batch_completion_memo = shared_search_memo
                    batch_state = None
                    batch_mask = None
                    batch_over_k = batch_open_count + 1 > self.max_concurrent_open_orders

                    if return_debug:
                        batch_state = self[batch_idx:batch_idx + 1]
                        batch_mask = batch_state.get_mask(skip_pickup_commitment=True)

                    for candidate_idx in candidate_index_list:
                        if batch_over_k:
                            pickup_commitment_mask[batch_idx, candidate_idx] = True
                            second_pickup_blocked += 1
                            if return_debug:
                                debug['diag_pickup_commitment_block_by_k'][batch_idx] += 1.0
                            continue

                        open_after_bits = batch_open_bits | (1 << candidate_idx)
                        candidate_is_passenger = batch_scalar_cache['passenger_orders'][candidate_idx]
                        candidate_scalar_cache = batch_scalar_cache
                        if candidate_is_passenger:
                            passenger_pickup_times_after = batch_passenger_pickup_times_before.clone()
                            passenger_pickup_times_after[candidate_idx] = batch_pickup_finish[candidate_idx]
                            candidate_scalar_cache = dict(batch_scalar_cache)
                            candidate_scalar_cache['pickup_times'] = list(batch_scalar_cache['pickup_times'])
                            candidate_scalar_cache['pickup_times'][candidate_idx] = float(batch_pickup_finish[candidate_idx].item())
                            candidate_scalar_cache['latest_delivery_arrival'] = list(batch_scalar_cache['latest_delivery_arrival'])
                            candidate_scalar_cache['latest_delivery_arrival'][candidate_idx] = min(
                                candidate_scalar_cache['pickup_times'][candidate_idx] + Config.PASSENGER_MAX_RIDE_TIME_MINUTES / 60.0,
                                candidate_scalar_cache['pickup_times'][candidate_idx] + candidate_scalar_cache['direct_ride_time'][candidate_idx] + Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES / 60.0,
                            ) if Config.HARD_PASSENGER_MAX_RIDE_TIME else None
                        else:
                            passenger_pickup_times_after = batch_passenger_pickup_times_before
                        start_node_key = self._pickup_node_key(candidate_idx)
                        completion_feasible, completion_block_reason = self._has_feasible_open_completion(
                            batch_pickup_coords[candidate_idx],
                            batch_pickup_finish[candidate_idx],
                            batch_trip_start_after_pickup,
                            None,
                            batch_delivery_coords,
                            batch_delivery_earliest,
                            batch_delivery_to_depot_time,
                            batch_passenger_orders,
                            passenger_pickup_times_after,
                            batch_direct_ride_time,
                            return_reason=return_debug,
                            ignore_trip_time=self.relax_pickup_commitment_trip_time,
                            memo=batch_completion_memo,
                            open_bits=open_after_bits,
                            start_node=start_node_key,
                            benchmark_stats=benchmark_stats,
                            scalar_cache=candidate_scalar_cache,
                        ) if return_debug else (
                            self._has_feasible_open_completion(
                                batch_pickup_coords[candidate_idx],
                                batch_pickup_finish[candidate_idx],
                                batch_trip_start_after_pickup,
                                None,
                                batch_delivery_coords,
                                batch_delivery_earliest,
                                batch_delivery_to_depot_time,
                                batch_passenger_orders,
                                passenger_pickup_times_after,
                                batch_direct_ride_time,
                                ignore_trip_time=self.relax_pickup_commitment_trip_time,
                                memo=batch_completion_memo,
                                open_bits=open_after_bits,
                                start_node=start_node_key,
                                benchmark_stats=benchmark_stats,
                                scalar_cache=candidate_scalar_cache,
                                ),
                            None,
                        )

                        next_state_feasible = True
                        fallback_safe = True
                        if completion_feasible and requires_post_pickup_delivery_check:
                            next_state_feasible, fallback_safe = self._evaluate_post_pickup_open_delivery(
                                batch_pickup_coords[candidate_idx],
                                batch_pickup_finish[candidate_idx],
                                batch_trip_start_after_pickup,
                                None,
                                batch_delivery_coords,
                                batch_delivery_earliest,
                                batch_delivery_to_depot_time,
                                batch_passenger_orders,
                                passenger_pickup_times_after,
                                batch_direct_ride_time,
                                memo=shared_search_memo,
                                open_bits=open_after_bits,
                                start_node=start_node_key,
                                benchmark_stats=benchmark_stats,
                                scalar_cache=candidate_scalar_cache,
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
                                    if batch_state is not None and batch_mask is not None:
                                        candidate_node = candidate_idx + 1
                                        selected = torch.tensor([candidate_node], dtype=torch.long, device=device)
                                        next_state = batch_state.update(selected, current_mask=batch_mask)
                                        next_state_reason = self._classify_next_state_delivery_block(next_state)
                                    else:
                                        next_state_reason = 'other'
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
                if return_debug and batch_open_count >= 1:
                    debug['diag_second_pickup_feasible'][batch_idx] = float(second_pickup_feasible)
                    debug['diag_second_pickup_blocked_by_commitment'][batch_idx] = float(second_pickup_blocked)

        commitment_mask_full = torch.zeros(batch_size, coords_active.size(1), dtype=torch.bool, device=device)
        if n_orders > 0:
            commitment_mask_full[:, 1:n_orders + 1] = pickup_commitment_mask
        before_mask = service_mask.clone()
        service_mask |= commitment_mask_full[:, 1:2 * n_orders + 1]
        if return_debug:
            debug['diag_mask_pickup_commitment'] = (service_mask & ~before_mask).sum(1).float()
        _record('mask_pickup_commitment', phase_start)

        phase_start = time.perf_counter() if benchmark_stats is not None else None
        if n_orders > 0 and self.enable_delivery_viability:
            delivery_viability_mask = torch.zeros(batch_size, n_orders, dtype=torch.bool, device=device)
            for batch_idx in range(batch_size):
                trip_start = current_time_flat[batch_idx] if int(prev_a_flat[batch_idx].item()) == 0 else trip_start_time_flat[batch_idx]
                scalar_cache = batch_scalar_caches[batch_idx]
                legal_orders, physical_orders, used_fallback = self._get_legal_delivery_orders(
                    cur_coord_flat[batch_idx],
                    current_time_flat[batch_idx],
                    trip_start,
                    open_mask_full[batch_idx],
                    delivery_coords[batch_idx],
                    delivery_earliest[batch_idx],
                    delivery_to_depot_time[batch_idx],
                    passenger_orders[batch_idx],
                    passenger_pickup_times_before[batch_idx],
                    direct_ride_time[batch_idx],
                    allow_fallback=self.enable_viability_fallback,
                    memo=shared_search_memo,
                    scalar_cache=scalar_cache,
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
        _record('mask_delivery_viability', phase_start)

        phase_start = time.perf_counter() if benchmark_stats is not None else None
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
        min_orders = max(int(self.min_orders_per_dispatch), 1)
        in_dispatch = (self.prev_a != 0)
        below_min_orders = (
            in_dispatch
            & (self.served_orders_since_dispatch < min_orders)
            & (~all_done.bool())
        ).squeeze(1)

        no_open_started_orders = (~self.has_open_started_orders()).squeeze(1)
        pickup_visited = self.visited_[:, :, 1:n_orders + 1].bool().squeeze(1)
        delivery_visited = self.visited_[:, :, n_orders + 1:2 * n_orders + 1].bool().squeeze(1)
        untouched_unrejected = (~pickup_visited) & (~delivery_visited) & (~self.rejected_.squeeze(1).bool())
        remaining_unrejected_orders = untouched_unrejected.sum(-1)
        at_depot_clean = (self.prev_a == 0).squeeze(1) & no_open_started_orders
        forbid_small_dispatch = at_depot_clean & (remaining_unrejected_orders > 0) & (remaining_unrejected_orders < min_orders)
        if forbid_small_dispatch.any():
            mask[forbid_small_dispatch, :, 1:2 * n_orders + 1] = True

        if return_debug:
            debug['diag_depot_min_orders_block'] = (below_min_orders | forbid_small_dispatch).float()
        mask[:, :, 0] = mask[:, :, 0] | below_min_orders[:, None].to(torch.uint8)
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

        reject_candidates = self._deterministic_reject_order(mask)
        reject_candidate_available = reject_candidates >= 0
        no_open_started_orders = (~self.has_open_started_orders()).squeeze(1)
        pre_departure_gate = (
            (self.prev_a == 0)
            & (self.used_capacity_passenger <= 1e-5)
            & (self.used_capacity_cargo <= 1e-5)
            & (~self.has_open_started_orders())
        ).squeeze(1)
        at_depot = (self.prev_a == 0).squeeze(1)
        depot_dead_end_reject_gate = at_depot & no_open_started_orders & all_order_nodes_masked.squeeze(-1).bool()
        inroute_dead_end_reject_gate = (
            (~at_depot)
            & below_min_orders
            & no_open_started_orders
            & all_order_nodes_masked.squeeze(-1).bool()
        )
        dead_end_reject_gate = depot_dead_end_reject_gate | inroute_dead_end_reject_gate
        reject_allowed = (
            reject_candidate_available
            & (~all_done.squeeze(-1).bool())
            & bool(self.allow_reject)
            & (pre_departure_gate | dead_end_reject_gate)
        )

        depot_cleanup_forces_reject = allow_depot_fallback.squeeze(-1).bool() & reject_allowed
        allow_depot_fallback = allow_depot_fallback & (~depot_cleanup_forces_reject).view(-1, 1)
        if return_debug:
            debug['diag_depot_fallback_used'] = allow_depot_fallback.squeeze(-1).float()
        mask[:, :, 0] = mask[:, :, 0] & (~allow_depot_fallback).to(torch.uint8)
        if depot_cleanup_forces_reject.any():
            mask[:, :, 0] = mask[:, :, 0] | depot_cleanup_forces_reject.view(-1, 1).to(torch.uint8)

        mask[:, :, reject_index] = (~reject_allowed).view(-1, 1).to(torch.uint8)
        if return_debug:
            reject_available = mask[:, :, reject_index].eq(0).view(-1)
            debug['diag_reject_candidate_available'] = reject_candidate_available.float()
            debug['diag_reject_allowed'] = reject_allowed.float()
            debug['diag_reject_predeparture_available'] = (reject_available & pre_departure_gate).float()
            debug['diag_reject_inroute_available'] = (reject_available & (~pre_departure_gate)).float()
            debug['diag_reject_dead_end_inroute_available'] = (reject_available & inroute_dead_end_reject_gate).float()
        _record('mask_finalize', phase_start)

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

        is_delivery = (actual_selected >= n_orders + 1) & (actual_selected <= 2 * n_orders) & (~is_reject)
        served_increment = is_delivery.to(self.served_orders_since_dispatch.dtype)
        new_served_orders_since_dispatch = self.served_orders_since_dispatch + served_increment
        depot_return = (actual_selected == DEPOT) & (~is_reject)
        new_served_orders_since_dispatch = torch.where(
            leaving_depot | depot_return,
            torch.zeros_like(new_served_orders_since_dispatch),
            new_served_orders_since_dispatch,
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
            update_mask = (order_one_hot.bool() & passenger_pickup_selected.unsqueeze(-1)).to(passenger_pickup_time.dtype)
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
            served_orders_since_dispatch=new_served_orders_since_dispatch,
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
                min_orders_per_dispatch=self.min_orders_per_dispatch,
                enable_delivery_viability=self.enable_delivery_viability,
                enable_viability_fallback=self.enable_viability_fallback,
                relax_pickup_commitment_trip_time=self.relax_pickup_commitment_trip_time,
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                deadlock_count=self.deadlock_count[key],
                deadlock_limit=self.deadlock_limit[key],
                served_orders_since_dispatch=self.served_orders_since_dispatch[key],
                terminal_=self.terminal_[key],
                i=self.i[key],
            )
        raise TypeError(f"Invalid key type: {type(key)}")
