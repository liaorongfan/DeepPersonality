from dpcv.tools.registry import Registry

DATALOADER_REGISTRY = Registry("DATALOADER")


def build_dataloader(cfg):
    name = cfg.DATALOADER.NAME
    dataloader = DATALOADER_REGISTRY.get(name)
    return dataloader
