# animal session
script/extract_test_pred_data.sh \
    config/unified_frame_images/15_multi_modal_pred.yaml \
    results/unified_frame_images/15_multi_modal_pred/09-09_01-19/checkpoint_72.pkl \
    multi_modal_output/impression/frame
script/extract_test_pred_data.sh \
    config/unified_face_images/8_multi_modal_pred_face.yaml \
    results/unified_face_images/8_multi_modal_pred/09-12_15-01/checkpoint_41.pkl \
    multi_modal_output/impression/face

echo " ======================== animal session output ================================="
python script/cmp_multi_modal_post_fusion.py \
    -v multi_modal_output/impression/frame/pred.pkl \
    -a multi_modal_output/impression_frame6/audio/pred.pkl \
    -o multi_modal_output/impression/face/pred.pkl \
    -l multi_modal_output/impression/frame/label.pkl \

























