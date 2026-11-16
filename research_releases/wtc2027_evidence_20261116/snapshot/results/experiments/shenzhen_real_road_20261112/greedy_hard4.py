"""Deterministic matrix-agnostic insertion start with a hard four-order trip floor."""


def insertion_options(engine, routes, order, existing_only=False):
    choices = []
    upper = len(routes) if existing_only else len(routes) + (len(routes) < engine.k)
    for route_id in range(upper):
        old = routes[route_id] if route_id < len(routes) else ()
        for pickup_pos in range(len(old) + 1):
            first = old[:pickup_pos] + (order,) + old[pickup_pos:]
            for delivery_pos in range(pickup_pos + 1, len(first) + 1):
                new = first[:delivery_pos] + (order + engine.n,) + first[delivery_pos:]
                if engine.route(new) is None:
                    continue
                trial = routes.copy()
                if route_id == len(routes):
                    trial.append(new)
                else:
                    trial[route_id] = new
                score = engine.cost(trial, strict=False)
                if score is not None:
                    choices.append((score["objective"], route_id, new))
    return sorted(choices, key=lambda value: (value[0], value[1], value[2]))


def make_start(engine):
    windows = engine.s["time_windows"]
    routes = []
    for order in sorted(range(1, engine.n + 1), key=lambda value: (windows[value - 1][1], value)):
        choices = insertion_options(engine, routes, order)
        if choices:
            _, route_id, new = choices[0]
            if route_id == len(routes):
                routes.append(new)
            else:
                routes[route_id] = new
    dropped = [order for route in routes if len(route) // 2 < 4 for order in route if order <= engine.n]
    routes = [route for route in routes if len(route) // 2 >= 4]
    for order in sorted(dropped):
        choices = insertion_options(engine, routes, order, existing_only=True)
        if choices:
            _, route_id, new = choices[0]
            routes[route_id] = new
    if engine.cost(routes) is None:
        raise ValueError("Insertion start did not satisfy the frozen hard constraints")
    return routes

