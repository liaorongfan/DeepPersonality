from exps_second_stage.feature_extract import feature_extract_true_personality
from dpcv.data.datasets.feature_extract_true_personality_dataset import (
    set_true_personality_dataloader,
    set_audiovisual_true_personality_dataloader,
    set_persemon_true_personality_dataloader,
    set_crnet_true_personality_dataloader,
)


if __name__ == "__main__":

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/lego_session/unified_frame_images/10_deep_bimodal_regression.yaml",
    #     model_weight="results_true_personality/unified_frame_images_lego/10_deep_bimodal_regression/06-29_18-37/checkpoint_0.pkl",
    #     data_loader=set_true_personality_dataloader,
    #     output_dir="datasets/second_stage_TP/deep_bimodal_reg_extract",
    # )

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/lego_session/unified_frame_images/08_interpert_img.yaml",
    #     model_weight="results_true_personality/unified_frame_images_lego/08_interpret_img/06-29_10-31/checkpoint_1.pkl",
    #     data_loader=set_true_personality_dataloader,
    #     output_dir="datasets/second_stage_TP/interpret_img",
    # )

    feature_extract_true_personality(
        cfg_file="config/true_personality/lego_session/unified_frame_images/01_senet.yaml",
        model_weight="results_true_personality/unified_frame_images_lego/01_senet/04-30_17-12/checkpoint_3.pkl",
        data_loader=set_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/senet",
    )

    feature_extract_true_personality(
        cfg_file="config/true_personality/lego_session/unified_frame_images/02_hrnet.yaml",
        model_weight="results_true_personality/unified_frame_images_lego/02_hrnet/04-30_19-28/checkpoint_1.pkl",
        data_loader=set_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/hrnet",
    )

    feature_extract_true_personality(
        cfg_file="config/true_personality/lego_session/unified_frame_images/03_swin_transformer.yaml",
        model_weight="results_true_personality/unified_frame_images_lego/03_swin_transformer/04-30_23-30/checkpoint_2.pkl",
        data_loader=set_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/swin_transformer",
    )

    feature_extract_true_personality(
        cfg_file="config/true_personality/lego_session/unified_frame_images/11_bimodal_resnet18.yaml",
        model_weight="results_true_personality/unified_frame_images_lego/11_bimodal_resnet18/07-03_00-03/checkpoint_0.pkl",
        data_loader=set_audiovisual_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/bimodal_resnet18",
    )

    feature_extract_true_personality(
        cfg_file="config/true_personality/lego_session/unified_frame_images/09_persemon.yaml",
        model_weight="results_true_personality/unified_frame_images_lego/09_persemon/06-29_16-39/checkpoint_1.pkl",
        data_loader=set_persemon_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/persemon",
    )

    feature_extract_true_personality(
        cfg_file="config/true_personality/lego_session/unified_frame_images/12_crnet.yaml",
        model_weight="results_true_personality/unified_frame_images_lego/12_crnet/06-30_20-57/checkpoint_2.pkl",
        data_loader=set_crnet_true_personality_dataloader,
        output_dir="datasets/second_stage_TP/crnet",
    )
