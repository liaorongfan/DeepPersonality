from dpcv.tools.registry import Registry

SOLVER_REGISTRY = Registry("SOLVER")


def build_solver(cfg, model):
    name = cfg.SOLVER.NAME
    solver = SOLVER_REGISTRY.get(name)
    return solver(cfg, model)


def build_scheduler(cfg, optimizer):
    name = cfg.SOLVER.SCHEDULER
    schedule = SOLVER_REGISTRY.get(name)
    return schedule(cfg, optimizer)
