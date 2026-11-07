def get_mask(self, return_debug=False, skip_pickup_commitment=False, benchmark_stats=None, return_urgency_slack=False, aux_outputs=None):
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
    if return_debug:
        before_mask = service_mask.clone()
    if n_orders > 0:
        service_mask[:, :n_orders] |= rejected
        service_mask[:, n_orders:] |= (pickup_not_done | rejected)
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
    if return_debug:
        before_mask = service_mask.clone()
    service_mask |= cap_mask_p
    if return_debug:
        debug['diag_mask_cap_passenger'] = (service_mask & ~before_mask).sum(1).float()

    cap_mask_c = (node_type_real == 0) & (demand_c_real > remaining_cap_c + 1e-5)
    if return_debug:
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
    pickup_tw_slack_hours = None
    is_passenger_node = node_type_active == 1
    is_cargo_node = node_type_active == 0
    is_pickup_node = (node_positions >= 1) & (node_positions <= n_orders)

    pickup_hard_gate = (
        (Config.HARD_PASSENGER_PICKUP_TIMEWINDOW & is_passenger_node)
        | (Config.HARD_CARGO_PICKUP_TIMEWINDOW & is_cargo_node)
    )
    pickup_hard_mask_full = pickup_hard_gate & is_pickup_node & (arrival_time > tw_end + 1e-5)
    if return_urgency_slack and n_orders > 0:
        pickup_tw_slack_hours = tw_end[:, 1:n_orders + 1] - arrival_time[:, 1:n_orders + 1]
    if return_urgency_slack and pickup_tw_slack_hours is None:
        pickup_tw_slack_hours = torch.zeros(batch_size, n_orders, device=device, dtype=arrival_time.dtype)
    if return_debug:
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
        pickup_to_delivery_time = self.pickup_to_delivery_time_static[ids_flat]
        direct_ride_time = pickup_to_delivery_time[:, order_indices, order_indices]
        excess_ride_time_hours = torch.clamp(ride_time_hours - direct_ride_time, min=0.0)
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
        if return_debug:
            before_delivery = service_mask[:, n_orders:2 * n_orders].clone()
        service_mask[:, n_orders:2 * n_orders] |= ride_time_violation
        if return_debug:
            debug['diag_mask_ride_time'] = (service_mask[:, n_orders:2 * n_orders] & ~before_delivery).sum(1).float()
    else:
        direct_ride_time = torch.zeros(batch_size, 0, device=device)
        passenger_orders = node_type_active[:, 1:1]
        delivery_indices = torch.zeros(0, dtype=torch.long, device=device)
        if return_debug:
            debug['diag_mask_ride_time'] = torch.zeros(batch_size, device=device)
    _record('mask_ride_time', phase_start)

    phase_start = time.perf_counter() if benchmark_stats is not None else None
    if Config.HARD_MAX_TRIP_TIME:
        depot_coord = coords_active[:, 0:1, :]
        dist_back = (coords_active - depot_coord).norm(p=2, dim=-1) * self.AREA_SIZE
        time_back = dist_back / self.VEHICLE_SPEED
        ongoing_trip_total = (arrival_time + self.SERVICE_TIME + time_back) - self.trip_start_time
        fresh_trip_total = travel_time + self.SERVICE_TIME + time_back
        predicted_total = torch.where(self.prev_a == 0, fresh_trip_total, ongoing_trip_total)
        trip_block = predicted_total[:, 1:2 * n_orders + 1] > self.MAX_TRIP_TIME + 1e-5
        if return_debug:
            before_service = service_mask.clone()
        service_mask |= trip_block
        if return_debug:
            debug['diag_mask_trip_time'] = (service_mask & ~before_service).sum(1).float()
    elif return_debug:
        debug['diag_mask_trip_time'] = torch.zeros(batch_size, device=device)
    _record('mask_trip_time', phase_start)

    phase_start = time.perf_counter() if benchmark_stats is not None else None
    if Config.HARD_OPERATION_END:
        if not Config.HARD_MAX_TRIP_TIME:
            depot_coord = coords_active[:, 0:1, :]
            dist_back = (coords_active - depot_coord).norm(p=2, dim=-1) * self.AREA_SIZE
            time_back = dist_back / self.VEHICLE_SPEED
        predicted_finish = arrival_time + self.SERVICE_TIME + time_back
        ops_block = predicted_finish[:, 1:2 * n_orders + 1] > self.OPERATION_END + 1e-5
        if return_debug:
            before_service = service_mask.clone()
        service_mask |= ops_block
        if return_debug:
            debug['diag_mask_ops_end'] = (service_mask & ~before_service).sum(1).float()
    elif return_debug:
        debug['diag_mask_ops_end'] = torch.zeros(batch_size, device=device)
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
        # Candidate-specific departure: depot waiting is outside the trip.
        candidate_departure = torch.maximum(
            self.current_time,
            time_windows_active[:, 1:n_orders + 1, 0] - travel_time[:, 1:n_orders + 1],
        )
        trip_start_after_pickup = torch.where(
            self.prev_a == 0, candidate_departure, self.trip_start_time.expand_as(candidate_departure)
        )
        passenger_pickup_times_before = self.passenger_pickup_time.squeeze(1)
        current_time_flat = self.current_time.squeeze(1)
        trip_start_time_flat = self.trip_start_time.squeeze(1)
        prev_a_flat = self.prev_a.squeeze(1)
        cur_coord_flat = self.cur_coord.squeeze(1)
        open_mask_full = open_started_mask
        shared_search_memo = {}
        batch_scalar_caches = None
        requires_scalar_cache = (not skip_pickup_commitment) or self.enable_delivery_viability
        if requires_scalar_cache:
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
                batch_scalar_cache = batch_scalar_caches[batch_idx]
                batch_completion_memo = shared_search_memo
                batch_state = None
                batch_mask = None
                batch_over_k = batch_open_count + 1 > self.max_concurrent_open_orders
                passenger_total_limit = Config.PASSENGER_MAX_RIDE_TIME_MINUTES / 60.0 if Config.HARD_PASSENGER_MAX_RIDE_TIME else None
                passenger_excess_limit = Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES / 60.0 if Config.HARD_PASSENGER_MAX_RIDE_TIME else None

                candidate_tensor = candidate_indices
                candidate_trip_start = batch_trip_start_after_pickup[candidate_tensor]
                candidate_pickup_finish = batch_pickup_finish[candidate_tensor]
                candidate_direct_ride_time = batch_direct_ride_time[candidate_tensor]
                candidate_arrival = candidate_pickup_finish + candidate_direct_ride_time
                candidate_service_start = torch.maximum(
                    candidate_arrival,
                    batch_delivery_earliest[candidate_tensor],
                )
                candidate_finish_with_return = (
                    candidate_service_start + self.SERVICE_TIME + batch_delivery_to_depot_time[candidate_tensor]
                )

                blocked_by_ride = torch.zeros_like(candidate_pickup_finish, dtype=torch.bool)
                blocked_by_trip = torch.zeros_like(candidate_pickup_finish, dtype=torch.bool)
                blocked_by_ops = torch.zeros_like(candidate_pickup_finish, dtype=torch.bool)

                candidate_is_passenger = batch_scalar_cache['passenger_orders'][candidate_tensor]
                candidate_latest_arrival = torch.full_like(candidate_pickup_finish, -1.0)
                if Config.HARD_PASSENGER_MAX_RIDE_TIME:
                    candidate_latest_arrival = candidate_pickup_finish + torch.minimum(
                        torch.full_like(candidate_direct_ride_time, passenger_total_limit),
                        candidate_direct_ride_time + passenger_excess_limit,
                    )
                    blocked_by_ride = candidate_is_passenger & (candidate_arrival > candidate_latest_arrival + 1e-5)

                if Config.HARD_MAX_TRIP_TIME:
                    blocked_by_trip = (
                        candidate_finish_with_return - candidate_trip_start
                    ) > self.MAX_TRIP_TIME + 1e-5

                if Config.HARD_OPERATION_END:
                    blocked_by_ops = candidate_finish_with_return > self.OPERATION_END + 1e-5

                # 必要条件：pickup 后直接送达自身 delivery 都不可行，则任何更长路径也不可行。
                candidate_necessary_block = blocked_by_ride | blocked_by_trip | blocked_by_ops
                self_candidate_necessary_block = candidate_necessary_block
                grouped_ride_reason = torch.zeros_like(candidate_necessary_block)
                grouped_trip_reason = torch.zeros_like(candidate_necessary_block)
                grouped_ops_reason = torch.zeros_like(candidate_necessary_block)
                if batch_open_count > 0:
                    candidate_open_mask = open_before[batch_idx].unsqueeze(0).expand(candidate_tensor.size(0), -1).clone()
                    candidate_open_mask[torch.arange(candidate_tensor.size(0), device=device), candidate_tensor] = True
                    candidate_open_positions = torch.nonzero(candidate_open_mask, as_tuple=False)
                    candidate_indices_by_pos = candidate_open_positions[:, 0]
                    open_order_indices = candidate_open_positions[:, 1]
                    start_node_keys = candidate_tensor + 1
                    travel_lookup_stack = batch_scalar_cache['travel_time_to_delivery_matrix'].index_select(0, start_node_keys)
                    grouped_candidate_ride = torch.zeros(candidate_tensor.size(0), dtype=torch.bool, device=device)
                    grouped_candidate_trip = torch.zeros(candidate_tensor.size(0), dtype=torch.bool, device=device)
                    grouped_candidate_ops = torch.zeros(candidate_tensor.size(0), dtype=torch.bool, device=device)
                    if candidate_open_positions.numel() > 0:
                        grouped_pickup_finish = candidate_pickup_finish[candidate_indices_by_pos]
                        grouped_arrival_lb = grouped_pickup_finish + travel_lookup_stack[candidate_indices_by_pos, open_order_indices]
                        grouped_latest_arrival = batch_scalar_cache['latest_delivery_arrival'][open_order_indices]
                        if Config.HARD_PASSENGER_MAX_RIDE_TIME:
                            same_order = open_order_indices == candidate_tensor[candidate_indices_by_pos]
                            grouped_latest_arrival = torch.where(
                                same_order,
                                candidate_latest_arrival[candidate_indices_by_pos],
                                grouped_latest_arrival,
                            )
                            ride_hits = (grouped_latest_arrival >= 0.0) & (grouped_arrival_lb > grouped_latest_arrival + 1e-5)
                            if ride_hits.any():
                                grouped_candidate_ride[candidate_indices_by_pos[ride_hits]] = True

                        grouped_service_start_lb = torch.maximum(
                            grouped_arrival_lb,
                            batch_scalar_cache['delivery_earliest'][open_order_indices],
                        )
                        grouped_finish_with_return_lb = grouped_service_start_lb + self.SERVICE_TIME + batch_scalar_cache['delivery_to_depot_time'][open_order_indices]
                        if Config.HARD_MAX_TRIP_TIME and (not self.relax_pickup_commitment_trip_time):
                            trip_hits = grouped_finish_with_return_lb - candidate_trip_start[candidate_indices_by_pos] > self.MAX_TRIP_TIME + 1e-5
                            if trip_hits.any():
                                grouped_candidate_trip[candidate_indices_by_pos[trip_hits]] = True
                        if Config.HARD_OPERATION_END:
                            ops_hits = grouped_finish_with_return_lb > self.OPERATION_END + 1e-5
                            if ops_hits.any():
                                grouped_candidate_ops[candidate_indices_by_pos[ops_hits]] = True

                    remaining_candidates = ~self_candidate_necessary_block
                    grouped_ride_reason = remaining_candidates & grouped_candidate_ride
                    remaining_candidates = remaining_candidates & (~grouped_ride_reason)
                    grouped_trip_reason = remaining_candidates & grouped_candidate_trip
                    remaining_candidates = remaining_candidates & (~grouped_trip_reason)
                    grouped_ops_reason = remaining_candidates & grouped_candidate_ops
                    candidate_necessary_block = self_candidate_necessary_block | grouped_ride_reason | grouped_trip_reason | grouped_ops_reason
                    if benchmark_stats is not None:
                        ride_pruned = int(grouped_ride_reason.sum().item())
                        trip_pruned = int(grouped_trip_reason.sum().item())
                        ops_pruned = int(grouped_ops_reason.sum().item())
                        if ride_pruned > 0:
                            self._record_benchmark_count(benchmark_stats, 'pc_pruned_by_ride_time', ride_pruned)
                        if trip_pruned > 0:
                            self._record_benchmark_count(benchmark_stats, 'pc_pruned_by_trip_time', trip_pruned)
                        if ops_pruned > 0:
                            self._record_benchmark_count(benchmark_stats, 'pc_pruned_by_ops_end', ops_pruned)

                candidate_block_ride = blocked_by_ride | grouped_ride_reason
                candidate_block_trip = blocked_by_trip | grouped_trip_reason
                candidate_block_ops = blocked_by_ops | grouped_ops_reason

                if return_debug:
                    batch_state = self[batch_idx:batch_idx + 1]
                    batch_mask = batch_state.get_mask(skip_pickup_commitment=True)

                if batch_open_count == 0 and (not requires_post_pickup_delivery_check):
                    # Fast path (语义等价): 没有 open 订单时，pickup commitment 退化为“该订单单独闭环是否可行”。
                    blocked_candidates = candidate_necessary_block
                    pickup_commitment_mask[batch_idx, candidate_tensor] = blocked_candidates

                    blocked_count = int(blocked_candidates.sum().item())
                    second_pickup_blocked += blocked_count
                    second_pickup_feasible += len(candidate_index_list) - blocked_count

                    if return_debug and blocked_count > 0:
                        debug['diag_pickup_commitment_block_by_completion'][batch_idx] += float(blocked_count)
                        blocked_remaining = blocked_candidates
                        ride_count = int((blocked_remaining & blocked_by_ride).sum().item())
                        blocked_remaining = blocked_remaining & (~blocked_by_ride)
                        trip_count = int((blocked_remaining & blocked_by_trip).sum().item())
                        blocked_remaining = blocked_remaining & (~blocked_by_trip)
                        ops_count = int((blocked_remaining & blocked_by_ops).sum().item())
                        other_count = int((blocked_remaining & (~blocked_by_ops)).sum().item())

                        if ride_count > 0:
                            debug['diag_pickup_commitment_block_by_completion_ride_time'][batch_idx] += float(ride_count)
                        if trip_count > 0:
                            debug['diag_pickup_commitment_block_by_completion_trip_time'][batch_idx] += float(trip_count)
                        if ops_count > 0:
                            debug['diag_pickup_commitment_block_by_completion_ops_end'][batch_idx] += float(ops_count)
                        if other_count > 0:
                            debug['diag_pickup_commitment_block_by_completion_other'][batch_idx] += float(other_count)
                    continue

                for candidate_pos, candidate_idx in enumerate(candidate_index_list):
                    candidate_trip_start_time = batch_trip_start_after_pickup[candidate_idx]
                    if batch_over_k:
                        pickup_commitment_mask[batch_idx, candidate_idx] = True
                        second_pickup_blocked += 1
                        if return_debug:
                            debug['diag_pickup_commitment_block_by_k'][batch_idx] += 1.0
                        continue

                    if bool(candidate_necessary_block[candidate_pos].item()):
                        pickup_commitment_mask[batch_idx, candidate_idx] = True
                        second_pickup_blocked += 1
                        if return_debug:
                            debug['diag_pickup_commitment_block_by_completion'][batch_idx] += 1.0
                            if bool(candidate_block_ride[candidate_pos].item()):
                                debug['diag_pickup_commitment_block_by_completion_ride_time'][batch_idx] += 1.0
                            elif bool(candidate_block_trip[candidate_pos].item()):
                                debug['diag_pickup_commitment_block_by_completion_trip_time'][batch_idx] += 1.0
                            elif bool(candidate_block_ops[candidate_pos].item()):
                                debug['diag_pickup_commitment_block_by_completion_ops_end'][batch_idx] += 1.0
                            else:
                                debug['diag_pickup_commitment_block_by_completion_other'][batch_idx] += 1.0
                        continue

                    open_after_bits = batch_open_bits | (1 << candidate_idx)
                    start_node_key = self._pickup_node_key(candidate_idx)
                    candidate_pickup_finish_value = float(batch_pickup_finish[candidate_idx].item())
                    candidate_has_override = bool(candidate_is_passenger[candidate_pos].item())
                    if candidate_has_override:
                        self._scalar_cache_set_candidate_pickup_override(
                            batch_scalar_cache,
                            candidate_idx,
                            candidate_pickup_finish_value,
                            float(candidate_latest_arrival[candidate_pos].item()) if Config.HARD_PASSENGER_MAX_RIDE_TIME else -1.0,
                        )

                    try:
                        if return_debug:
                            completion_feasible, completion_block_reason = self._has_feasible_open_completion(
                                batch_pickup_coords[candidate_idx],
                                batch_pickup_finish[candidate_idx],
                                candidate_trip_start_time,
                                None,
                                batch_delivery_coords,
                                batch_delivery_earliest,
                                batch_delivery_to_depot_time,
                                batch_passenger_orders,
                                batch_passenger_pickup_times_before,
                                batch_direct_ride_time,
                                return_reason=True,
                                ignore_trip_time=self.relax_pickup_commitment_trip_time,
                                memo=batch_completion_memo,
                                open_bits=open_after_bits,
                                start_node=start_node_key,
                                benchmark_stats=benchmark_stats,
                                scalar_cache=batch_scalar_cache,
                            )
                        else:
                            completion_feasible = self._has_feasible_open_completion(
                                batch_pickup_coords[candidate_idx],
                                batch_pickup_finish[candidate_idx],
                                candidate_trip_start_time,
                                None,
                                batch_delivery_coords,
                                batch_delivery_earliest,
                                batch_delivery_to_depot_time,
                                batch_passenger_orders,
                                batch_passenger_pickup_times_before,
                                batch_direct_ride_time,
                                ignore_trip_time=self.relax_pickup_commitment_trip_time,
                                memo=batch_completion_memo,
                                open_bits=open_after_bits,
                                start_node=start_node_key,
                                benchmark_stats=benchmark_stats,
                                scalar_cache=batch_scalar_cache,
                            )
                            completion_block_reason = None

                        next_state_feasible = True
                        fallback_safe = True
                        if completion_feasible and requires_post_pickup_delivery_check:
                            next_state_feasible, fallback_safe = self._evaluate_post_pickup_open_delivery(
                                batch_pickup_coords[candidate_idx],
                                batch_pickup_finish[candidate_idx],
                                candidate_trip_start_time,
                                None,
                                batch_delivery_coords,
                                batch_delivery_earliest,
                                batch_delivery_to_depot_time,
                                batch_passenger_orders,
                                batch_passenger_pickup_times_before,
                                batch_direct_ride_time,
                                memo=shared_search_memo,
                                open_bits=open_after_bits,
                                start_node=start_node_key,
                                benchmark_stats=benchmark_stats,
                                scalar_cache=batch_scalar_cache,
                            )
                    finally:
                        if candidate_has_override:
                            self._scalar_cache_clear_candidate_pickup_override(batch_scalar_cache)

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

    if return_debug:
        before_mask = service_mask.clone()
    if n_orders > 0:
        service_mask[:, :n_orders] |= pickup_commitment_mask
    if return_debug:
        debug['diag_mask_pickup_commitment'] = (service_mask & ~before_mask).sum(1).float()
    _record('mask_pickup_commitment', phase_start)

    phase_start = time.perf_counter() if benchmark_stats is not None else None
    if n_orders > 0 and self.enable_delivery_viability:
        delivery_viability_mask = torch.zeros(batch_size, n_orders, dtype=torch.bool, device=device)
        for batch_idx in range(batch_size):
            trip_start = current_time_flat[batch_idx] if int(prev_a_flat[batch_idx].item()) == 0 else trip_start_time_flat[batch_idx]
            scalar_cache = batch_scalar_caches[batch_idx]
            legal_bitmap, physical_bitmap, used_fallback = self._get_legal_delivery_orders(
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
                start_node=self._current_start_node_key(prev_a_flat[batch_idx].item()),
                scalar_cache=scalar_cache,
                return_bitmap=True,
            )
            if not physical_bitmap.any().item():
                continue
            blocked = physical_bitmap & (~legal_bitmap)
            if blocked.any():
                delivery_viability_mask[batch_idx] |= blocked
            if used_fallback and return_debug:
                debug['diag_delivery_viability_fallback'][batch_idx] = 1.0
        if return_debug:
            before_viability = service_mask.clone()
        service_mask[:, n_orders:2 * n_orders] |= delivery_viability_mask
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

    no_open_started_orders = open_started_count == 0
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
    pre_departure_gate = (
        (self.prev_a == 0)
        & (self.used_capacity_passenger <= 1e-5)
        & (self.used_capacity_cargo <= 1e-5)
        & no_open_started_orders.view(-1, 1)
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

    if return_urgency_slack and aux_outputs is not None:
        aux_outputs['pickup_tw_slack_hours'] = pickup_tw_slack_hours
    if return_debug:
        if return_urgency_slack:
            debug['pickup_tw_slack_hours'] = pickup_tw_slack_hours
        return mask.bool(), debug
    return mask.bool()
