"""Matrix-based scalar route evaluator for the Shenzhen road experiment."""
import math
from collections import OrderedDict


def _python(value):
    return value.detach().cpu().tolist() if hasattr(value, "detach") else value


class RoadRouteEngine:
    def __init__(self, sample, config, profile, vehicles=None):
        self.s = {key: _python(value) for key, value in sample.items()}
        self.c = dict(config)
        self.profile = dict(profile)
        self.n = len(self.s["loc"]) // 2
        self.k = vehicles or math.ceil(self.n / 6)
        self.distance = self.s["road_distance_matrix_km"]
        self.travel = self.s["road_travel_time_matrix_hours"]
        expected = 2 * self.n + 1
        if len(self.distance) != expected or len(self.travel) != expected:
            raise ValueError("Road matrices must include depot plus every pickup and delivery")
        for matrix in (self.distance, self.travel):
            if any(len(row) != expected for row in matrix):
                raise ValueError("Road matrices must be square")
            if any(not math.isfinite(float(v)) or float(v) < 0 for row in matrix for v in row):
                raise ValueError("Road matrices must be finite and nonnegative")
        self.cache = OrderedDict()
        self.evaluations = 0
        self.hits = 0

    def route(self, nodes):
        key = tuple(nodes)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.evaluations += 1
        result = self._route(key)
        self.cache[key] = result
        if len(self.cache) > 100000:
            self.cache.popitem(last=False)
        return result

    def _route(self, nodes):
        if not nodes:
            return dict(orders=(), energy=0.0, distance=0.0, passenger_minutes=0.0, cargo_minutes=0.0)
        c, s, n, tol = self.c, self.s, self.n, 1e-4
        if any(not isinstance(x, int) or not 1 <= x <= 2 * n for x in nodes):
            return None
        depart = max(c["OPERATION_START"], s["time_windows"][nodes[0] - 1][0] - self.travel[0][nodes[0]])
        t, cur = depart, 0
        lp = lc = energy = distance = pd = cd = 0.0
        picked, done, opened = set(), set(), {}
        for node in nodes:
            j, order, pickup = node - 1, (node - 1) % n + 1, node <= n
            km = self.distance[cur][node]
            arrival = t + self.travel[cur][node]
            load_kg = lp * c["PASSENGER_CAPACITY"] * c["PASSENGER_WEIGHT_KG"] + lc * c["CARGO_CAPACITY"] * c["CARGO_UNIT_WEIGHT_KG"]
            energy += km * c["ENERGY_BASE"] * (1 + load_kg / c["ENERGY_WEIGHT_REF"]) * c["ELECTRICITY_PRICE"]
            distance += km
            start = max(arrival, s["time_windows"][j][0])
            finish = start + c["SERVICE_TIME"]
            if pickup:
                if order in picked:
                    return None
                if (s["node_type"][j] == 1 or c["HARD_CARGO_PICKUP_TIMEWINDOW"]) and start > s["time_windows"][j][1] + tol:
                    return None
                picked.add(order)
                opened[order] = finish
            else:
                if order not in opened or order in done:
                    return None
                if s["node_type"][j] == 1:
                    ride = start - opened[order]
                    direct = self.travel[order][node]
                    if ride > c["PASSENGER_MAX_RIDE_TIME_MINUTES"] / 60 + tol or ride - direct > c["PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES"] / 60 + tol:
                        return None
                    pd += max(0.0, arrival - s["time_windows"][j][1]) * 60
                else:
                    cd += max(0.0, arrival - s["time_windows"][j][1]) * 60
                del opened[order]
                done.add(order)
            lp += s["demand_passenger"][j]
            lc += s["demand_cargo"][j]
            if not -tol <= lp <= 1 + tol or not -tol <= lc <= 1 + tol or len(opened) > 6 or finish > c["OPERATION_END"] + tol:
                return None
            cur, t = node, finish
        if opened or picked != done:
            return None
        km = self.distance[cur][0]
        returned = t + self.travel[cur][0]
        if returned > c["OPERATION_END"] + tol or returned - depart > c["MAX_TRIP_TIME"] + tol:
            return None
        load_kg = lp * c["PASSENGER_CAPACITY"] * c["PASSENGER_WEIGHT_KG"] + lc * c["CARGO_CAPACITY"] * c["CARGO_UNIT_WEIGHT_KG"]
        energy += km * c["ENERGY_BASE"] * (1 + load_kg / c["ENERGY_WEIGHT_REF"]) * c["ELECTRICITY_PRICE"]
        distance += km
        return dict(orders=tuple(sorted(done)), energy=energy, distance=distance, passenger_minutes=pd, cargo_minutes=cd)

    def cost(self, routes, strict=True):
        active = [tuple(route) for route in routes if route]
        metrics = [self.route(route) for route in active]
        if len(active) > self.k or any(value is None for value in metrics):
            return None
        ids = [order for value in metrics for order in value["orders"]]
        if len(set(ids)) != len(ids) or (strict and any(len(value["orders"]) < 4 for value in metrics)):
            return None
        c, p = self.c, self.profile
        energy = sum(value["energy"] for value in metrics)
        cp = sum(value["passenger_minutes"] for value in metrics) * c["PASSENGER_DELAY_COST"]
        minutes = sum(value["cargo_minutes"] for value in metrics)
        b1, b2 = c["CARGO_DELAY_TIER1_MIN"], c["CARGO_DELAY_TIER2_MIN"]
        cc = min(minutes, b1) * c["CARGO_DELAY_COST"] + max(0.0, min(minutes, b2) - b1) * c["CARGO_DELAY_COST_30_60"] + max(0.0, minutes - b2) * c["CARGO_DELAY_COST_60_PLUS"]
        vc = len(active) * c["VEHICLE_COST"]
        rp = (self.n - len(ids)) * c["ALPHA_REJECT"]
        objective = c["ALPHA_ENERGY"] * energy / p["energy_cost_raw"] + c["ALPHA_DELAY"] * (cp / p["passenger_delivery_delay_cost_raw"] + cc / p["cargo_delay_cost_raw"]) + c["ALPHA_VEHICLE"] * vc / max(1.0, p["vehicle_cost_norm"]) + rp
        return dict(objective=objective, raw_cost=energy + cp + cc + vc + rp, energy=energy, distance=sum(value["distance"] for value in metrics), passenger_delay_cost=cp, cargo_delay_cost=cc, vehicle_cost=vc, reject_penalty=rp, completed=len(ids), vehicles=len(active))

    def actions(self, routes):
        served = sum(len(route) // 2 for route in routes)
        return [node for route in routes if route for node in list(route) + [0]] + [2 * self.n + 1] * (self.n - served)

