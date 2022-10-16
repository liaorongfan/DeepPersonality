from exps_second_stage.feature_extract import feature_extract_true_personality
from dpcv.data.datasets.feature_extract_true_personality_dataset import (
    set_true_personality_dataloader,
    set_audiovisual_true_personality_dataloader,
    set_persemon_true_personality_dataloader,
    set_crnet_true_personality_dataloader,
)


if __name__ == "__main__":

    feature_extract_true_personality(
        cfg_file="config/true_personality/ghost_session/unified_frame_images/10_deep_bimodal_regression.yaml",
        model_weight="results_true_personality/unified_frame_images_ghost/10_deep_bimodal_regression.yaml/06-27_05-42/checkpoint_0.pkl",
        data_loader=set_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/deep_bimodal_reg_extract",
    )

    feature_extract_true_personality(
        cfg_file="config/true_personality/ghost_session/unified_frame_images/08_interpert_img.yaml",
        model_weight="results_true_personality/unified_frame_images_ghost/08_interpret_img.yaml/06-26_11-56/checkpoint_1.pkl",
        data_loader=set_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/interpret_img",
    )

    feature_extract_true_personality(
        cfg_file="config/true_personality/ghost_session/unified_frame_images/01_setnet.yaml",
        model_weight="results_true_personality/unified_frame_images_ghost/01_senet/04-26_10-09/checkpoint_3.pkl",
        data_loader=set_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/senet",
    )

    feature_extract_true_personality(
        cfg_file="config/true_personality/ghost_session/unified_frame_images/02_hrnet.yaml",
        model_weight="results_true_personality/unified_frame_images_ghost/02_hrnet/04-29_03-50/checkpoint_0.pkl",
        data_loader=set_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/hrnet",
    )

    feature_extract_true_personality(
        cfg_file="config/true_personality/ghost_session/unified_frame_images/03_swin_transformer.yaml",
        model_weight="results_true_personality/unified_frame_images_ghost/03_swin_transformer/04-29_21-16/checkpoint_0.pkl",
        data_loader=set_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/swin_transformer",
    )

    feature_extract_true_personality(
        cfg_file="config/true_personality/ghost_session/unified_frame_images/11_bimodal_resnet18.yaml",
        model_weight="results_true_personality/unified_frame_images_ghost/11_bimodal_resnet18/06-27_15-48/checkpoint_0.pkl",
        data_loader=set_audiovisual_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/bimodal_resnet18",
    )

    feature_extract_true_personality(
        cfg_file="config/true_personality/ghost_session/unified_frame_images/09_persemon.yaml",
        model_weight="results_true_personality/unified_frame_images_ghost/09_persemon/06-27_00-55/checkpoint_0.pkl",
        data_loader=set_persemon_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/persemon",
    )

    feature_extract_true_personality(
        cfg_file="config/true_personality/ghost_session/unified_frame_images/12_crnet.yaml",
        model_weight="results_true_personality/unified_frame_images_ghost/12_crnet/06-27_20-06/checkpoint_2.pkl",
        data_loader=set_crnet_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/crnet",
    )
