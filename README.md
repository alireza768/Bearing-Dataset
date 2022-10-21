# Conrod-Bearing Dataset

- collected from the top view of a 4-cylinder EF7 engine block
- engine block contains four moving bearings
- contains 2000 images in 16 classes
- image resolution has been set to 3280×2460
- Invalve all wrong assembly of conrod-bearing
- Using 8-megapixel Raspberry Pi camera to collect the images
- annotate the data in the YOLO format

# Pre Processing

**Traditional data augmentation**
- Rotate
- Flip 
- Noise

**Deeplearning based filters**
- Dexi-Ned
- Phase Steretch Transform

# Approches

**Classification**
- Architecture Selection
- Transfer Learning
- Dexi-Ned / Phase Steretch Transform
- Traditional data augmentation

**Object Detection**
- Images Annotation
- Architecture Selection

**Segmentation**
- Images Annotation
- Architecture Selection

# Setup
```
Data
 |-- checkpoints
 |  |-- Classification 
 |  |  |-- Xception model.h5
 |  |-- Segmentation 
 |  |  |-- segmentation model.h5
 |-- Classification Data
 |  |-- Classification Train
 |  |  |-- 1
 |  |  |-- 2
 |  |  |-- 3
 |  |  |-- 4
 |  |  |-- 5
 |  |  |-- 6
 |  |  |-- 7
 |  |  |-- 8
 |  |  |-- 9
 |  |  |-- 10
 |  |  |-- 11
 |  |  |-- 12
 |  |  |-- 13
 |  |  |-- 14
 |  |  |-- 15
 |  |  |-- 16
 |  |-- Classification Validation
 |  |  |-- 1
 |  |  |-- 2
 |  |  |-- 3
 |  |  |-- 4
 |  |  |-- 5
 |  |  |-- 6
 |  |  |-- 7
 |  |  |-- 8
 |  |  |-- 9
 |  |  |-- 10
 |  |  |-- 11
 |  |  |-- 12
 |  |  |-- 13
 |  |  |-- 14
 |  |  |-- 15
 |  |  |-- 16
 |  |-- Classification Test
 |-- Segmentation Data
 |  |-- segmentation Train
 |  |  |-- Images
 |  |  |-- Images Masks
 |  |-- segmentation Validation
 |  |  |-- Images
 |  |  |-- Images Masks
 |  |-- segmentation Test
 ...
```
## Usage
