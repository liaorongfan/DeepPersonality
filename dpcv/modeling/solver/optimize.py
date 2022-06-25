import torch.optim as optim
from .build import SOLVER_REGISTRY


@SOLVER_REGISTRY.register()
def sgd(cfg, model):
    return optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.LR_INIT,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )


@SOLVER_REGISTRY.register()
def crnet_solver(cfg, model):
    solver = cfg.SOLVER
    optimizer_fir = optim.SGD(
        model.parameters(), lr=solver.LR_INIT, momentum=solver.MOMENTUM, weight_decay=solver.WEIGHT_DECAY
    )
    optimizer_sec = optim.Adam(
        model.parameters(), betas=(solver.BETA_1, solver.BETA_2), lr=solver.LR_INIT, weight_decay=solver.WEIGHT_DECAY
    )
    return [optimizer_fir, optimizer_sec]


@SOLVER_REGISTRY.register()
def adam(cfg, model):
    return optim.Adam(model.parameters(), lr=cfg.SOLVER.LR_INIT,)
