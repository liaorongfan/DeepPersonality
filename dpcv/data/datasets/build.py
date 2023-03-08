from dpcv.tools.registry import Registry

DATA_LOADER_REGISTRY = Registry("DATA_LOADER")


def build_dataloader(cfg):
    name = cfg.DATA_LOADER.NAME
    dataloader = DATA_LOADER_REGISTRY.get(name)
    dataset_name = cfg.DATA_LOADER.DATASET

    if dataset_name:
        dataset = DATA_LOADER_REGISTRY.get(dataset_name)

        if not cfg.TEST.TEST_ONLY:
            data_loader_dicts = {
                "train": dataloader(cfg, dataset, mode="train"),
                "valid": dataloader(cfg, dataset, mode="valid"),
                "test": dataloader(cfg, dataset, mode="test"),
            }
        else:
            data_loader_dicts = {
                "test": dataloader(cfg, dataset, mode="test"),
            }

    else:
        if not cfg.TEST.TEST_ONLY:
            data_loader_dicts = {
                "train": dataloader(cfg, mode="train"),
                "valid": dataloader(cfg, mode="valid"),
                "test": dataloader(cfg, mode="test"),
            # "full_test": dataloader(cfg, mode="full_test")
            }
        else:
            data_loader_dicts = {
                "test": dataloader(cfg, mode="test"),
            }
        if cfg.TEST.FULL_TEST:
            data_loader_dicts["full_test"] = dataloader(cfg, mode="full_test")
    return data_loader_dicts
