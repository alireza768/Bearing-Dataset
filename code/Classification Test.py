
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
i = 51
while True:

 def load_image(img_path, show=True):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,axis=0 )
  #  print (img_tensor)        # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.
    #print (img_tensor)                                  # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


 if __name__ == "__main__":

    # load model

    model = load_model('D:/model/Best Xeption model.h5')
    model.summary()

    # image path

    img_path = 'D:/Data/Classification Validation/13/Bearing_ '+str(i)+'.jpg'
    i += 1

    # load a single image
    new_image = load_image(img_path)

    # check prediction
    pred = model.predict(new_image)
    classes_x=np.argmax(pred,axis=1)
    print(classes_x)