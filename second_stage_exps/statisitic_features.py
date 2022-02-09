from dpcv.checkpoint.save import load_model
from dpcv.config.default_config_opt import cfg, cfg_from_file
from dpcv.experiment.exp_runner import ExpRunner


def feature_extract(cfg_file, model_weight):

    cfg_from_file(cfg_file)
    runner = ExpRunner(cfg)
    runner.model = load_model(runner.model.cuda(), model_weight)
    ocean_acc_avg, ocean_acc, dataset_output, dataset_label = runner.trainer.full_test(
        runner.data_loader["full_test"], runner.model
    )
    print(ocean_acc_avg, ocean_acc)


if __name__ == "__main__":
    import os
    # print("current file", __file__)
    os.chdir("..")
    feature_extract(
        cfg_file="config/unified_frame_images/06_interpret_cnn.yaml",
        model_weight="results/unified_frame_images/06_interpret_cnn/checkpoint_interpret-cnn_acc_9118.pkl",
    )

