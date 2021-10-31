from datetime import datetime
from dpcv.checkpoint.save import save_model, resume_training, load_model
from dpcv.evaluation.metrics import compute_pcc, compute_ccc


def run(cfg, data_loader, model, loss_f, optimizer, scheduler, trainer, collector, logger, log_dir):

    if cfg.TEST_ONLY:
        logger.info("Test only mode")
        model = load_model(model, cfg.WEIGHT)
        ocean_acc_avg, ocean_acc, dataset_output, dataset_label = trainer.test(data_loader["test"], model)
        logger.info(f"average acc of OCEAN:{ocean_acc},\taverage acc [{ocean_acc_avg}]")
        if cfg.COMPUTE_PCC:
            pcc_dict = compute_pcc(dataset_output, dataset_label)
            logger.info(f"pcc and p_value:\n{pcc_dict}")
        if cfg.COMPUTE_CCC:
            ccc_dict = compute_ccc(dataset_output, dataset_label)
            logger.info(f"ccc:\n{ccc_dict}")
        return

    if cfg.RESUME:
        model, optimizer, epoch = resume_training(cfg.RESUME, model, optimizer)
        cfg.START_EPOCH = epoch
        logger.info(f"resume training from {cfg.RESUME}")

    for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
        trainer.train(data_loader["train"], model, loss_f, optimizer, epoch)
        trainer.valid(data_loader["valid"], model, loss_f, epoch)
        scheduler.step()

        if collector.model_save:
            save_model(epoch, collector.best_valid_acc, model, optimizer, log_dir, cfg)
            collector.update_best_epoch(epoch)

    collector.draw_epo_info(cfg.MAX_EPOCH - cfg.START_EPOCH, log_dir)
    logger.info(
        "{} done, best acc: {} in :{}".format(
            datetime.strftime(datetime.now(), '%m-%d_%H-%M'), collector.best_valid_acc, collector.best_epoch
        )
    )




