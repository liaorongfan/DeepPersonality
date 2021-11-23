from dpcv.tools.registry import Registry

DATA_LOADER_REGISTRY = Registry("DATA_LOADER")


def build_dataloader(cfg):
    name = cfg.DATA_LOADER.NAME
    dataloader = DATA_LOADER_REGISTRY.get(name)
    return dataloader
