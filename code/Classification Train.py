import os
from tensorflow import keras
import tensorflow as tf
base_model = keras.applications.Xception(
      weights="imagenet",  # Load weights pre-trained on ImageNet.
      input_shape=(71, 71, 3),
      include_top=False,
    )  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = True

# Create new model on top
inputs = keras.Input(shape=(71, 71, 3))
#x = data_augmentation(inputs)  # Apply random data augmentation
x = base_model(inputs, training=True)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(2,activation='softmax')(x)
model = keras.Model(inputs, outputs)
model.summary()
train_dir = ('D:/Data/Classification Train')
validation_dir = ('D:/Data/Classification Validation')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(71, 71),
    batch_size=1,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(71, 71),
    batch_size=1,
    class_mode='categorical')


model.compile(
    optimizer=keras.optimizers.Adam(1e-4),  # Low learning rate
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy()],
)

checkpoint_path = 'D:/model/Best Xeption model.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)
# metric = 'val_accuracy'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='val_categorical_accuracy',
    mode='max',
    verbose=2,
    save_best_only=True)

epochs = 100
hist = model.fit(train_generator, epochs=epochs, callbacks=[model_checkpoint_callback],
                 validation_data=validation_generator)
import matplotlib.pyplot as plt
plt.plot(hist.history["categorical_accuracy"])
plt.plot(hist.history["val_categorical_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper left")
plt.show()