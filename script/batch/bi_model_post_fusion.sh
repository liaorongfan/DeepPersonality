# Talk session
#script/extract_test_pred_data.sh \
#    config/audio/talk_session/01_aud_linear_regressor.yaml \
#    results_true_personality/audio_talk/01_aud_linear_regressor/05-05_22-19/checkpoint_92.pkl \
#    pose_fuse_bimodal_reg/talk/audio
#script/extract_test_pred_data.sh \
#    config/true_personality/talk_session/unified_frame_images/10_deep_bimodal_regression.yaml \
#    results_true_personality/unified_frame_images_talk/10_deep_bimodal_regression/06-22_20-01/checkpoint_0.pkl \
#    pose_fuse_bimodal_reg/talk/visual
#
#echo " ========================= talk session output =================================="
#python script/cmp_multi_modal_post_fusion.py \
#    -v pose_fuse_bimodal_reg/talk/visual/pred.pkl \
#    -a pose_fuse_bimodal_reg/talk/audio/pred.pkl \
#    -l pose_fuse_bimodal_reg/talk/audio/label.pkl \


# animal session
script/extract_test_pred_data.sh \
    config/audio/talk_session/01_aud_linear_regressor.yaml \
    results_true_personality/audio_talk/01_aud_linear_regressor/05-05_22-19/checkpoint_92.pkl \
    pose_fuse_bimodal_reg/animal/audio
script/extract_test_pred_data.sh \
    config/true_personality/talk_session/unified_frame_images/10_deep_bimodal_regression.yaml \
    results_true_personality/unified_frame_images_talk/10_deep_bimodal_regression/06-22_20-01/checkpoint_0.pkl \
    pose_fuse_bimodal_reg/animal/visual

echo " ======================== animal session output ================================="
python script/cmp_multi_modal_post_fusion.py \
    -v pose_fuse_bimodal_reg/talk/visual/pred.pkl \
    -a pose_fuse_bimodal_reg/talk/audio/pred.pkl \
    -l pose_fuse_bimodal_reg/talk/audio/label.pkl \


# ghost session
script/extract_test_pred_data.sh \
    config/audio/talk_session/01_aud_linear_regressor.yaml \
    results_true_personality/audio_talk/01_aud_linear_regressor/05-05_22-19/checkpoint_92.pkl \
    pose_fuse_bimodal_reg/ghost/audio
script/extract_test_pred_data.sh \
    config/true_personality/talk_session/unified_frame_images/10_deep_bimodal_regression.yaml \
    results_true_personality/unified_frame_images_talk/10_deep_bimodal_regression/06-22_20-01/checkpoint_0.pkl \
    pose_fuse_bimodal_reg/ghost/visual

echo " ========================== ghost session output ================================="
python script/cmp_multi_modal_post_fusion.py \
    -v pose_fuse_bimodal_reg/talk/visual/pred.pkl \
    -a pose_fuse_bimodal_reg/talk/audio/pred.pkl \
    -l pose_fuse_bimodal_reg/talk/audio/label.pkl \


# lego session
script/extract_test_pred_data.sh \
    config/audio/talk_session/01_aud_linear_regressor.yaml \
    results_true_personality/audio_talk/01_aud_linear_regressor/05-05_22-19/checkpoint_92.pkl \
    pose_fuse_bimodal_reg/lego/audio
script/extract_test_pred_data.sh \
    config/true_personality/talk_session/unified_frame_images/10_deep_bimodal_regression.yaml \
    results_true_personality/unified_frame_images_talk/10_deep_bimodal_regression/06-22_20-01/checkpoint_0.pkl \
    pose_fuse_bimodal_reg/lego/visual
echo " =========================== lego session output ==============================="
python script/cmp_multi_modal_post_fusion.py \
    -v pose_fuse_bimodal_reg/talk/visual/pred.pkl \
    -a pose_fuse_bimodal_reg/talk/audio/pred.pkl \
    -l pose_fuse_bimodal_reg/talk/audio/label.pkl \
