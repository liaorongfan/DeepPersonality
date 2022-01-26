from dpcv.tools.registry import Registry

TRANSFORM_REGISTRY = Registry("TRANSFORM_OPT")


def build_transform_spatial(cfg):
    name = cfg.DATA_LOADER.TRANSFORM
    transform = TRANSFORM_REGISTRY.get(name)
    return transform()
