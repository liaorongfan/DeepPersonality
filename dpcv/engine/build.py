from dpcv.tools.registry import Registry

TRAINER_REGISTRY = Registry("TRAINER")


def build_trainer(cfg, collector, logger):
    name = cfg.TRAINER.NAME
    trainer_cls = TRAINER_REGISTRY.get(name)
    return trainer_cls(cfg, collector, logger)
