import os
from dpcv.checkpoint.save import load_model
from dpcv.config.default_config_opt import cfg, cfg_from_file
from dpcv.data.datasets.feature_extract_dataset import (
    setup_dataloader, setup_crnet_dataloader,
    setup_persemon_dataloader, setup_bimodal_resnet_dataloader,
)
from dpcv.data.datasets.feature_extract_true_personality_dataset import (
    set_true_personality_dataloader,
    set_audiovisual_true_personality_dataloader,
    set_persemon_true_personality_dataloader,
    set_crnet_true_personality_dataloader,
)
from dpcv.experiment.exp_runner import ExpRunner


def feature_extract(cfg_file, model_weight, data_loader, output_dir, return_feat=False):

    cfg_from_file(cfg_file)
    cfg.MODEL.RETURN_FEATURE = return_feat

    runner = ExpRunner(cfg)
    runner.model = load_model(runner.model, model_weight)
    # ocean_acc_avg, ocean_acc, dataset_output, dataset_label = runner.trainer.full_test(
    #     data_loader(cfg, mode="test"), runner.model
    # )
    # print(ocean_acc_avg, ocean_acc)

    for mode in ["train", "valid", "test"]:
        # note if cuda out of memory, run each mode separately
        dataloader = data_loader(cfg, mode=mode)
        output_sub_dir = os.path.join(output_dir, mode)
        runner.data_extract(dataloader, output_sub_dir)


def feature_extract_true_personality(cfg_file, model_weight, data_loader, output_dir, return_feat=False):
    cfg_from_file(cfg_file)
    cfg.MODEL.RETURN_FEATURE = return_feat

    runner = ExpRunner(cfg, feature_extract=True)
    runner.model = load_model(runner.model, model_weight)

    for mode in ["train", "valid", "test"]:
        # note if cuda out of memory, run each mode separately
        dataloader = data_loader(cfg, mode=mode)
        output_sub_dir = os.path.join(output_dir, mode, f"{cfg.DATA.SESSION}_{mode}")
        runner.data_extract(dataloader, output_sub_dir)


if __name__ == "__main__":

    # # interpret_cnn feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/10_swin_transformer.yaml",
    #     model_weight="results/unified_frame_images/10_swin_transformer/12-13_21-28/checkpoint_110.pkl",
    #     data_loader=setup_dataloader,
    #     output_dir="datasets/second_stage/swin_transformer_extract",
    # )

    # # interpret_cnn feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/09_hrnet.yaml",
    #     model_weight="results/unified_frame_images/09_hrnet/checkpoint_hrnet_acc_905.pkl",
    #     data_loader=setup_dataloader,
    #     output_dir="datasets/second_stage/hrnet_extract",
    # )


    # # interpret_cnn feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/08_senet.yaml",
    #     model_weight="results/unified_frame_images/08_senet/checkpoint_senet_acc_9051.pkl",
    #     data_loader=setup_dataloader,
    #     output_dir="datasets/second_stage/senet_extract",
    # )

    # interpret_cnn feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/06_interpret_cnn.yaml",
    #     model_weight="results/unified_frame_images/06_interpret_cnn/checkpoint_interpret-cnn_acc_9118.pkl",
    #     data_loader=setup_dataloader,
    #     output_dir="datasets/second_stage/interpret_cnn_extract",
    # )

    # # persemon feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/05_persemon.yaml",
    #     model_weight="results/unified_frame_images/05_peremon/12-23_00-07/checkpoint_160.pkl",
    #     data_loader=setup_persemon_dataloader,
    #     output_dir="datasets/second_stage/persemon_extract",
    # )

    # # crnet feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/04_crnet.yaml",
    #     model_weight="results/unified_frame_images/04_crnet/12-07_09-01/checkpoint_85.pkl",
    #     data_loader=setup_crnet_dataloader,
    #     output_dir="datasets/stage_two/cr_net_extract",
    # )

    # bimodal_renet18 feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/03_bimodal_resnet18.yaml",
    #     model_weight="results/unified_frame_images/03_bimodal_resnet/12-19_23-35/checkpoint_297.pkl",
    #     data_loader=setup_bimodal_resnet_dataloader,
    #     output_dir="datasets/second_stage/bimodal_resnet18_extract",
    # )

    # deep_bimodal_regression feature extract
    feature_extract(
        cfg_file="config/unified_frame_images/01_deep_bimodal_regression.yaml",
        model_weight="results/unified_frame_images/01_deep_bimodal/12-06_00-50/checkpoint_84.pkl",
        data_loader=setup_dataloader,
        output_dir="datasets/second_stage/deep_bimodal_reg_extract",
    )

