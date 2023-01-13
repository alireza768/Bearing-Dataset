# Conrod bearing dataset

- Being collected from bottom view of four-cylinder engine blocks
- There are 2 to the power of 4 (means 16) different bearing installation cases (called classes)
- Containing 2000 images in each 16-class
- Using an 8MP Raspberry Pi camera for image capture
- Setting the resolution of images on 3280×2460 pixels
- Using 8MP Raspberry Pi camera to collect the images
- Annotating the images in YOLO format
- Download [**(Classification Data)**](https://drive.google.com/file/d/1x1fWg54HHkBc4zABBs3n2Szl6izrwr3n/view?usp=sharing)
- Download [**(Detection Data)**](https://drive.google.com/file/d/13qtMvgaqP61M0iQkpjxUt4VBJAKlkvl8/view?usp=sharing)
- Download [**(Segmentation Data)**](https://drive.google.com/file/d/1AxQGpTHrd4rRwLRwhj3ROJuT0lWSpeG5/view?usp=sharing)

![Capture](https://user-images.githubusercontent.com/85845544/197382474-270632ca-1a53-483b-abfa-61344cb1d571.JPG)

# Preprocessing

**Traditional data augmentation**
- 15° clockwise rotation
- 15° counterclockwise rotation
- 30° clockwise rotation
- 30° counterclockwise rotation
- Horizontal flip
- Noise
- Gaussian filter

**Deep learning-based filters**
- Dexi-Ned
- Phase Stretch Transform

# Approach

**Classification**
- Architecture selection
- Transfer learning
- Dexi-Ned / Phase Stretch Transform
- Traditional data augmentation

**Segmentation**
- Annotation of images
- Architecture selection

**Detection**
- Annotation of images
- Architecture selection

![Capture1](https://user-images.githubusercontent.com/85845544/197391026-5b557bc0-319d-435d-b1e0-bedb894362fd.PNG)

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
        ...
        ...
        ...
 |  |  |-- 16
 |  |-- Classification Validation
 |  |  |-- 1
 |  |  |-- 2
 |  |  |-- 3
        ...
        ...
        ...
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
 |  |  |-- test image
 |  |  |-- test image mask
 ...
```
## Usage

**Classification**
- Download Xception model.h5 and put it in "Checkpoints" folder [**(Xception model)**](https://drive.google.com/file/d/1pkuIa-d7a8mNGxbwka7QeBu-W3zoBXpZ/view?usp=sharing)
- Put test image in "Classification Test" folder
- Run Classification Test.py

**Segmentation**
- Download Segmentation model.h5 and put it in "Checkpoints" folder [**(Segmentation model)**](https://drive.google.com/file/d/1Lgp7sLMFQNq0uQMpmch66KbsrDpPzbk_/view?usp=sharing)
- Put test image in "Segmentation Test" folder
- Run Segmentation Test.py

**Detection**

- Download "Yolov4" folder and put it in Google Drive [**(Yolov4)**](https://drive.google.com/drive/folders/1EDUZ6yi2qUP65OGfx7cfDpPRSNAvPrPe?usp=sharing)
- Create a "Bearing_test_images" folder in Google Drive and put a custom image in this folder
- Run Detection Test.ipynb

![predictions ](https://user-images.githubusercontent.com/85845544/197379493-e1580868-cd68-471b-ba76-e1334bfe0647.jpg)

## Real-time industrial application of classification approach

- Using Raspberry Pi4 8GB board
- Using 8MP v2 Raspberry Pi camera
- Using trained .h5 file in classification approach
- Using TensorFlow Lite on Raspberry Pi
- Using Raspberry Pi GPIO to show final results

![20220813_130507](https://user-images.githubusercontent.com/85845544/197379046-95c4e241-56b0-4b53-8c7b-b8fd0365ac75.jpg)
