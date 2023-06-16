# 64 frames
# python tools/run_exp.py -c config/impression/long_context/64_frames/04_resnet50_3d_face.yaml --resume results/unified_face_images/04_resnet50_3d_face/06-06_02-38/checkpoint_36.pkl --max_epoch 100 --lr 0.04
python tools/run_exp.py -c config/impression/long_context/64_frames/06_tpn_face.yaml --resume results/unified_face_images/13_tpn/06-06_22-10/checkpoint_1.pkl --max_epoch 100 --lr 0.001
# python tools/run_exp.py -c config/impression/long_context/64_frames/07_vat_face.yaml --resume results/unified_face_images/07_vat_face/06-09_03-37/checkpoint_38.pkl --max_epoch 100 --lr 0.004

# python tools/run_exp.py -c config/true_personality/long_context/04_resnet3d_face.yaml
# python tools/run_exp.py -c config/true_personality/long_context/06_tpn_face.yaml
# python tools/run_exp.py -c config/true_personality/long_context/07_vat_face_video_level.yaml