# Dataset

## Expected dataset structure for [flower-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/):
```
102flowers/
  train/
  valid/
  test/
  imagelabels.mat
```

## Expected dataset structure for [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/index.html):
```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
      # train.txt or val.txt, if you use these splits
  JPEGImages/
```