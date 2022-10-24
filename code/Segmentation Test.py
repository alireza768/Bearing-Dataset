import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#import tensorflow.compat.v2
#import tensorflow.compat
import cv2
#import keras
import numpy as np
import matplotlib.pyplot as plt
DATA_DIR = ('D:/Segmentation Data')
x_train_dir = os.path.join(DATA_DIR, 'segmentation Train/Images')
y_train_dir = os.path.join(DATA_DIR, 'segmentation Train/Images Masks')

x_valid_dir = os.path.join(DATA_DIR, 'segmentation Validation/Images')
y_valid_dir = os.path.join(DATA_DIR, 'segmentation Validation/Images Masks')

x_test_dir = os.path.join(DATA_DIR, 'segmentation Test/test image')
y_test_dir = os.path.join(DATA_DIR, 'segmentation Test/test image mask')

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class Dataset:

    CLASSES = ['healthy', 'defect']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
import keras

class Dataloder(keras.utils.Sequence):

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


import albumentations as A

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(480, 480)

    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

import segmentation_models as sm

BACKBONE = 'mobilenet'
BATCH_SIZE = 1
CLASSES = ['Healthy', 'Defect']
LR = 0.0001
EPOCHS = 0

preprocess_input = sm.get_preprocessing(BACKBONE)
import segmentation_models as sm

sm.set_framework('tf.keras')

sm.framework()


# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES)+1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'
#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

from tensorflow import keras
# define optomizer
optim = keras.optimizers.Adam(LR)
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 0.5, 2]))
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
#model.compile(optim, total_loss, metrics)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    classes=CLASSES,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
# load best weights

def my_custom_func():
    # your code
    'sigmoid' if n_classes == 1 else 'softmax'
    return
from keras.models import load_model
model = load_model('D:/checkpoints/Segmentation/segmentation model.h5', compile=False)
model.compile(optim, total_loss, metrics)
import cv2
q=480
while True:

 cap = cv2.VideoCapture(0)
 cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 ret, frame = cap.read()
 n = cv2.rotate(frame, cv2.ROTATE_180)
 #cv2.imwrite('first1.jpg', frame)
 #b = cv2.imread('first1.jpg')
 b = cv2.resize(b, (480, 480))#

 DATA_DIR = ('gggg')
 filename = 'D:/Data/Test/50.png'
 #cv2.imwrite(filename, b)
 cv2.imwrite('D:/200/' + str(q) + '.png', frame)
 q+=1
 cap.release()
 image, gt_mask = test_dataset[0]
 image = np.expand_dims(image, axis=0)
 pr_mask = model.predict(image)
 g = pr_mask[0, :, :, :]
 cv2.imwrite('h.jpg',q)
 m=cv2.imread('h.jpg')
 visualize(
        image=denormalize(image.squeeze()),
        gt_mask=gt_mask.squeeze(),
        pr_mask=pr_mask.squeeze(),
    )




