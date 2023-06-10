# 32 frames
# python tools/run_exp.py -c config/impression/long_context/04_resnet50_3d_face.yaml
# python tools/run_exp.py -c config/impression/long_context/06_tpn_face.yaml
# python tools/run_exp.py -c config/impression/long_context/07_vat_face.yaml

python tools/run_exp.py -c config/true_personality/long_context/04_resnet3d_face.yaml
python tools/run_exp.py -c config/true_personality/long_context/06_tpn_face.yaml
python tools/run_exp.py -c config/true_personality/long_context/07_vat_face_video_level.yaml