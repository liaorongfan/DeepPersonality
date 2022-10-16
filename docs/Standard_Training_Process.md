## Standard Training Process:
For fire comparison between different models, we set the control of initial data processing and model preparation
### Data Process
1. All frames in a video are resize to [w: 456, h: 256] by opencv-python
```python
frame = cv2.resize(frame, (456, 256), interpolation=cv2.INTER_CUBIC)
```
2. Faces are extracted and aligned from frames with resolution of [w: 112, h: 112] by dlib tools

```python
face = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")(frame)
face = rotate_face(face)
```
3. when training every video is down sampled to 100 consecutive frames around 5 frames per second
```python
num_img = len(imgs_ls)
sample_frames = np.linspace(0, num_img, 100, endpoint=False, dtype=np.int16)
```

4. Image augmentation operations remain the same for different experiments(models)  
```python
transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(0.5),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Model Initilization
1. All model weights initiate with a same way and don't load any pretrained weights.
```python
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
```