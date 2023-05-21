from dpcv.exps_second_stage.feature_extract import feature_extract_true_personality
from dpcv.data.datasets.feature_extract_dataset_tp import (
    set_true_personality_dataloader,
    set_vat_tp_dataloader,
    set_multi_modal_pred_tp_dataloader,
    set_crnet_aud_true_personality_dataloader,
    set_audiovisual_true_personality_dataloader,
    set_persemon_true_personality_dataloader,
    set_crnet_true_personality_dataloader,
)


if __name__ == "__main__":

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/animal_session/unified_frame_images/10_deep_bimodal_regression.yaml",
    #     model_weight="results_true_personality/unified_frame_images_animal/10_deep_bimodal_regression.yaml/06-20_16-59/checkpoint_2.pkl",
    #     data_loader=set_true_personality_dataloader,
    #     output_dir="datasets/second_stage_TP/deep_bimodal_reg_extract",
    # )

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/animal_session/unified_frame_images/08_interpert_img.yaml",
    #     model_weight="results_true_personality/unified_frame_images_animal/08_interpret_img/06-20_00-52/checkpoint_3.pkl",
    #     data_loader=set_true_personality_dataloader,
    #     output_dir="datasets/second_stage_TP/interpret_img",
    # )

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/animal_session/unified_frame_images/01_setnet.yaml",
    #     model_weight="results_true_personality/unified_frame_images_animal/01_senet/checkpoint_15.pkl",
    #     data_loader=set_true_personality_dataloader,
    #     output_dir="datasets/second_stage_TP/senet",
    # )

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/animal_session/unified_frame_images/02_hrnet.yaml",
    #     model_weight="results_true_personality/unified_frame_images_animal/02_hrnet/04-20_13-54/checkpoint_2.pkl",
    #     data_loader=set_true_personality_dataloader,
    #     output_dir="datasets/second_stage_TP/hrnet",
    # )

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/animal_session/unified_frame_images/03_swin_transformer.yaml",
    #     model_weight="results_true_personality/unified_frame_images_animal/03_swin_transformer/checkpoint_4.pkl",
    #     data_loader=set_true_personality_dataloader,
    #     output_dir="datasets/second_stage_TP/swin_transformer",
    # )

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/animal_session/unified_frame_images/11_bimodal_resnet18.yaml",
    #     model_weight="results_true_personality/unified_frame_images_animal/11_bimodal_resnet18/06-22_07-02/checkpoint_3.pkl",
    #     data_loader=set_audiovisual_true_personality_dataloader,
    #     output_dir="datasets/second_stage_TP/bimodal_resnet18",
    # )

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/animal_session/unified_frame_images/09_persemon.yaml",
    #     model_weight="results_true_personality/unified_frame_images_animal/09_persemon.yaml/06-22_01-18/checkpoint_0.pkl",
    #     data_loader=set_persemon_true_personality_dataloader,
    #     output_dir="datasets/second_stage_TP/persemon",
    # )

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/animal_session/unified_frame_images/12_crnet.yaml",
    #     model_weight="results_true_personality/unified_frame_images_animal/12_crnet/06-23_21-10/checkpoint_2.pkl",
    #     data_loader=set_crnet_true_personality_dataloader,
    #     output_dir="datasets/second_stage_TP/crnet",
    # )
# ---------------------------------------------------------------------------------------
    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/all_sessions/02_hrnet_face.yaml",
    #     model_weight="results_true_personality/all_sesstion/unified_face_images_all/02_hrnet/03-14_08-56/checkpoint_1.pkl",
    #     data_loader=set_true_personality_dataloader,
    #     output_dir="datasets/model_output_features/02_hrnet_face",
    #     return_feat=True,
    # )

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/all_sessions/12_crnet.yaml",
    #     model_weight="results_true_personality/all_sesstion/unified_face_images_all/12_crnet/03-18_03-38/checkpoint_2.pkl",
    #     data_loader=set_crnet_true_personality_dataloader,
    #     output_dir="datasets/model_output_features/crnet",
    #     return_feat=True,
    # )

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/all_sessions/03_aud_crnet.yaml",
    #     model_weight="results_true_personality/all_sesstion/audio_all/03_aud_crnet/03-19_00-21/checkpoint_68.pkl",
    #     data_loader=set_crnet_aud_true_personality_dataloader,
    #     output_dir="datasets/model_output_features/crnet-aud",
    #     return_feat=True,
    # )

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/all_sessions/07_vat_face_video_level.yaml",
    #     model_weight="results_true_personality/all_sesstion/unified_face_images_all/07_vat_video_level/03-16_10-42/checkpoint_1.pkl",
    #     data_loader=set_vat_tp_dataloader,
    #     output_dir="datasets/model_output_features/07_vat_face_video_level",
    #     return_feat=True,
    # ) 

    # feature_extract_true_personality(
    #     cfg_file="config/true_personality/all_sessions/14_multi_modal_pred_face.yaml",
    #     model_weight="results_true_personality/all_sesstion/unified_face_images_all/14_multi_modal_pred/03-21_00-06/checkpoint_1.pkl",
    #     data_loader=set_multi_modal_pred_tp_dataloader,
    #     output_dir="datasets/model_output_features/14_multi_modal_pred",
    #     return_feat=True,
    # )

    feature_extract_true_personality(
        cfg_file="config/true_personality/all_sessions/06_multi_modal_pred_audio.yaml",
        model_weight="results_true_personality/all_sesstion/audio_all/06_multi_modal_pred/03-19_15-44/checkpoint_1.pkl",
        data_loader=set_multi_modal_pred_tp_dataloader,
        output_dir="datasets/model_output_features/06_multi_modal_pred_audio",
        return_feat=True,
    )
