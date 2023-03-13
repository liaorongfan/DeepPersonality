cd /hy-tmp/animal
echo $PWD
unzip train_frame_face_1.zip -d animal_train
unzip train_frame_face_2.zip -d animal_train
unzip train_frame_face_3.zip -d animal_train
unzip train_frame_face_4.zip -d animal_train
unzip train_frame_face_5.zip -d animal_train
unzip train_frame_face_6.zip -d animal_train

cd /hy-tmp/ghost
echo $PWD

unzip ghost_test_frame_face.zip 
unzip ghost_train_frame_face.zip 
unzip ghost_valid_frame_face.zip 