from dpcv.tools.registry import Registry

LOSS_FUNC_REGISTRY = Registry("LOSS_FUNC")


def build_loss_func(cfg):
    name = cfg.LOSS.NAME
    loss_func = LOSS_FUNC_REGISTRY.get(name)
    return loss_func()
