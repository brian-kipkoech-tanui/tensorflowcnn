# Computer Vision Using Convolutional Neural Networks

- Studies of the visual cortex inspired the neocognitron, which gradually evolved into Convolutional Neural Networks.
- Why not use deep neural network with fully connected layers for image recognition tasks?
- * Breaks down for larger images because of the huge number of parameters it requires.
- * CNNs solves this problem using partially connected layers and weight sharing.

## Convolutional Layers

- Neurons in the first Convolutional layer are not connected to every single pixel in the input image, but only to pixels in their receprive field.
- Each neuron in the second convolutional layer is connected only to neurons located within a small rectangle in the first layer.
- This architecture allows the network to concentrate on small low-level features in the first hidden layer, then assemble them into larger high-level features in the nect hidden ayer, and so on.
- In order for a layer to have the same height and width as the previous layer, it is common to add zeros around the inputs (Zero Padding)
- The shift from one receprive field to the next is called a stride.
- Example:
> A 5 by 5 input layer could be connected to a 3 by 4 layer, using a 3 by 3 receptive fields and  a stride of 2.

## Filters/Convolution Kernels

- A layer full of neurons using the same filter outputs a feature map,which highlights the areas in an image that activate the filter the most.

## Stacking Multiple Feature Maps

- A convolutional layer has multiple filters and outputs one feature map per filter.
- A convolutional layer simultenously applies multiple trainable filters to its inputs, making it capable of detecting multiple features anywhere in its inputs.
- The fact that all neurons in a feature map share the same parameters dramatically reduces the number of parameters in the model.

- Once the CNN has learned to recognize a pattern in one location, it can recognize it in any other location.


```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.__version__
```




    '2.11.0'



# Get the data


```python
import requests

def extract_cifar_data(url, filename="cifar.tar.gz"):
    """A function for extracting the CIFAR-100 dataset and storing it as a gzipped file
    
    Arguments:
    url      -- the URL where the dataset is hosted
    filename -- the full path where the dataset will be written
    
    """
    
    # Todo: request the data from the data url
    # Hint: use `requests.get` method
    r = requests.get(url)
    with open(filename, "wb") as file_context:
        file_context.write(r.content)
    return
```


```python
# extract_cifar_data("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz") 
```


```python
# import tarfile

# with tarfile.open("cifar.tar.gz", "r:gz") as tar:
#     tar.extractall()
```


```python
# import pickle

# with open("./cifar-100-python/meta", "rb") as f:
#     dataset_meta = pickle.load(f, encoding='bytes')

# with open("./cifar-100-python/test", "rb") as f:
#     dataset_test = pickle.load(f, encoding='bytes')

# with open("./cifar-100-python/train", "rb") as f:
#     dataset_train = pickle.load(f, encoding='bytes')
```


```python
# # Feel free to explore the datasets

# dataset_train.keys()
```


```python
# import numpy as np

# # Each 1024 in a row is a channel (red, green, then blue)
# row = dataset_train[b'data'][6]
# red, green, blue = row[0:1024], row[1024:2048], row[2048:]

# # Each 32 items in the channel are a row in the 32x32 image
# red = red.reshape(32,32)
# green = green.reshape(32,32)
# blue = blue.reshape(32,32)

# # Combine the channels into a 32x32x3 image!
# combined = np.dstack((red,green,blue))
```


```python
# #  All in one:
# test_image = np.dstack((
#     row[0:1024].reshape(32,32),
#     row[1024:2048].reshape(32,32),
#     row[2048:].reshape(32,32)
# ))
```


```python
import matplotlib.pyplot as plt
plt.imshow(test_image);
plt.show()
```


```python
import pandas as pd

# Todo: Filter the dataset_train and dataset_meta objects to find the label numbers for Bicycle and Motorcycles
desired_label_nos = {x:idx for idx, x in enumerate(dataset_meta[b'fine_label_names']) if x==b'bicycle' or x==b'motorcycle'}
# Label numbers for bicycle and motorcycles
desired_label_nos
```


```python
df_train = pd.DataFrame({
    "filenames": dataset_train[b'filenames'],
    "labels": dataset_train[b'fine_labels'],
    "row": range(len(dataset_train[b'filenames']))
})

# Drop all rows from df_train where label is not 8 or 48
df_train_bike = df_train.loc[(df_train.labels==8),:]   #TODO: Fill in
df_train_motorbike = df_train.loc[(df_train.labels==48),:]

# Decode df_train.filenames so they are regular strings
df_train_bike["filenames"] = df_train_bike["filenames"].apply(
    lambda x: x.decode("utf-8")
)
df_train_motorbike["filenames"] = df_train_motorbike["filenames"].apply(
    lambda x: x.decode("utf-8")
)


df_test = pd.DataFrame({
    "filenames": dataset_test[b'filenames'],
    "labels": dataset_test[b'fine_labels'],
    "row": range(len(dataset_test[b'filenames']))
})

# Drop all rows from df_test where label is not 8 or 48
df_test_bike = df_test.loc[(df_test.labels==8),:]   #TODO: Fill in
df_test_motorbike = df_test.loc[(df_test.labels==48),:]

# Decode df_test.filenames so they are regular strings
df_test_bike["filenames"] = df_test_bike["filenames"].apply(
    lambda x: x.decode("utf-8")
)
df_test_motorbike["filenames"] = df_test_motorbike["filenames"].apply(
    lambda x: x.decode("utf-8")
)
```


```python
import os
def save_images(imgs_details_df, imgs_dataset, target_folderpath):
    '''
    This function takes the following inputs and saves all images to the 'target_folder' in the local directory
    inputs:
    imgs_details_df: DataFrame containing details of images like filenames, finelabels (specific image labels) 
                     and row number details 
    imgs_dataset   : Original CIFAR-100 images dataset containing image data in the row major form that further 
                     needs to be transformed into 3 channeled image array
    target_folder  : Output folder where the transformed images are saved
    '''
    
    # Saving dataset 'row' numbers of corresponding images from 'imgs_details_df' dataframe
    row_number_imgs_data = imgs_details_df['row'] 
    
    # Corresponding image filenames in the dataset
    imgs_file_names = imgs_details_df['filenames']
    
    for row, img_name in zip(row_number_imgs_data, imgs_file_names):
        
        #Grab the image data in row-major form
        img = imgs_dataset[b'data'][row]

        # Consolidated stacking/reshaping from earlier (For all 3-R,G,B channels)
        target = np.dstack((
                        img[0:1024].reshape(32,32,1),
                        img[1024:2048].reshape(32,32,1),
                        img[2048:].reshape(32,32,1)
                         ))
    
        # Save the image to target folder (local directory)
        try:
            save_img_filepath = os.path.join(target_folderpath, img_name)
            plt.imsave(save_img_filepath, target)
            
            # For printing the saved image filepath in correct format
            raw_save_img_filepath = r"{}".format(save_img_filepath)
            nraw_save_img_filepath = os.path.normpath(raw_save_img_filepath)
            print(f"Saved: {img_name} to {nraw_save_img_filepath}")
            
        # Return any signal data you want for debugging
        except RuntimeError as e:
            return f"Error: {e} \nAn error encountered while saving {img_name} file to the target folder located at {target_folderpath}"

## TODO: save ALL images using the save_images function

# Save all transformed train dataset images in "./train"
save_images(imgs_details_df= df_train_bike,
            imgs_dataset= dataset_train, 
            target_folderpath= "./dataset/train/bicycle")
save_images(imgs_details_df= df_train_motorbike,
            imgs_dataset= dataset_train, 
            target_folderpath= "./dataset/train/motorcycle")

# Save all transformed test dataset images in "./test"
save_images(imgs_details_df= df_test_bike,
            imgs_dataset= dataset_test, 
            target_folderpath= "./dataset/test/bicycle")
save_images(imgs_details_df= df_test_motorbike,
            imgs_dataset= dataset_test, 
            target_folderpath= "./dataset/test/motorcycle")
```

# Data Preprocessing

## Preprocessing the training set


```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    './dataset/train/',
    target_size=(32, 32),
    batch_size=32,
    class_mode='binary'
)
```

    Found 1000 images belonging to 2 classes.
    

## Preprocessing the Test Data


```python
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    './dataset/test',
    target_size=(32,32),
    batch_size=32,
    class_mode='binary'
)
```

    Found 200 images belonging to 2 classes.
    

# Building the cnn

## Initializing the cnn


```python
cnn = tf.keras.models.Sequential()
```

## Step1 - Convolution


```python
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu',
    input_shape=[32,32,3]
))
```

## Step2 - Pooling


```python
cnn.add(tf.keras.layers.MaxPool2D(
    pool_size=2,
    strides=2
))
```

## Adding a second Convolutional layer


```python
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu'
))
# add a maxpooling layer
cnn.add(tf.keras.layers.MaxPool2D(
    pool_size=2,
    strides=2
))
```

## Step 3 - Flattening


```python
cnn.add(tf.keras.layers.Flatten())
```

## Step 4 - Full Connection


```python
cnn.add(tf.keras.layers.Dense(
    units=128,
    activation='relu'
))
```

## Step 5 - Output layer


```python
cnn.add(tf.keras.layers.Dense(
    units=1,
    activation='sigmoid'
))
```

# Training the model

## Compiling the model


```python
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Training the cnn on the training set and evaluating it on the Test set


```python
cnn.fit(x=training_set, validation_data=test_set, epochs=100, callbacks=tf.keras.callbacks.EarlyStopping(patience=10))
```

    Epoch 1/100
    32/32 [==============================] - 5s 126ms/step - loss: 0.5860 - accuracy: 0.6920 - val_loss: 0.4746 - val_accuracy: 0.7300
    Epoch 2/100
    32/32 [==============================] - 2s 63ms/step - loss: 0.4558 - accuracy: 0.7890 - val_loss: 0.3888 - val_accuracy: 0.8100
    Epoch 3/100
    32/32 [==============================] - 2s 59ms/step - loss: 0.4423 - accuracy: 0.8110 - val_loss: 0.3374 - val_accuracy: 0.8650
    Epoch 4/100
    32/32 [==============================] - 2s 66ms/step - loss: 0.4104 - accuracy: 0.8020 - val_loss: 0.4320 - val_accuracy: 0.8000
    Epoch 5/100
    32/32 [==============================] - 2s 61ms/step - loss: 0.3466 - accuracy: 0.8440 - val_loss: 0.3710 - val_accuracy: 0.8500
    Epoch 6/100
    32/32 [==============================] - 2s 61ms/step - loss: 0.3637 - accuracy: 0.8320 - val_loss: 0.3130 - val_accuracy: 0.8600
    Epoch 7/100
    32/32 [==============================] - 2s 61ms/step - loss: 0.3560 - accuracy: 0.8540 - val_loss: 0.2954 - val_accuracy: 0.8900
    Epoch 8/100
    32/32 [==============================] - 2s 61ms/step - loss: 0.3562 - accuracy: 0.8450 - val_loss: 0.3263 - val_accuracy: 0.8650
    Epoch 9/100
    32/32 [==============================] - 2s 64ms/step - loss: 0.3222 - accuracy: 0.8530 - val_loss: 0.4871 - val_accuracy: 0.7750
    Epoch 10/100
    32/32 [==============================] - 2s 72ms/step - loss: 0.3126 - accuracy: 0.8720 - val_loss: 0.4397 - val_accuracy: 0.8150
    Epoch 11/100
    32/32 [==============================] - 2s 66ms/step - loss: 0.3123 - accuracy: 0.8700 - val_loss: 0.4090 - val_accuracy: 0.8300
    Epoch 12/100
    32/32 [==============================] - 2s 64ms/step - loss: 0.3134 - accuracy: 0.8580 - val_loss: 0.2854 - val_accuracy: 0.8750
    Epoch 13/100
    32/32 [==============================] - 2s 66ms/step - loss: 0.2942 - accuracy: 0.8750 - val_loss: 0.2861 - val_accuracy: 0.8750
    Epoch 14/100
    32/32 [==============================] - 2s 63ms/step - loss: 0.3150 - accuracy: 0.8630 - val_loss: 0.4066 - val_accuracy: 0.8450
    Epoch 15/100
    32/32 [==============================] - 2s 66ms/step - loss: 0.3163 - accuracy: 0.8700 - val_loss: 0.2786 - val_accuracy: 0.8800
    Epoch 16/100
    32/32 [==============================] - 2s 64ms/step - loss: 0.2732 - accuracy: 0.8830 - val_loss: 0.2746 - val_accuracy: 0.8850
    Epoch 17/100
    32/32 [==============================] - 2s 69ms/step - loss: 0.2728 - accuracy: 0.8820 - val_loss: 0.3645 - val_accuracy: 0.8650
    Epoch 18/100
    32/32 [==============================] - 2s 73ms/step - loss: 0.2585 - accuracy: 0.8800 - val_loss: 0.2603 - val_accuracy: 0.9000
    Epoch 19/100
    32/32 [==============================] - 2s 70ms/step - loss: 0.2583 - accuracy: 0.8950 - val_loss: 0.2707 - val_accuracy: 0.8950
    Epoch 20/100
    32/32 [==============================] - 2s 61ms/step - loss: 0.2779 - accuracy: 0.8710 - val_loss: 0.2798 - val_accuracy: 0.8850
    Epoch 21/100
    32/32 [==============================] - 2s 66ms/step - loss: 0.2425 - accuracy: 0.9000 - val_loss: 0.3311 - val_accuracy: 0.8650
    Epoch 22/100
    32/32 [==============================] - 2s 63ms/step - loss: 0.2365 - accuracy: 0.9000 - val_loss: 0.3341 - val_accuracy: 0.8750
    Epoch 23/100
    32/32 [==============================] - 2s 60ms/step - loss: 0.2445 - accuracy: 0.8980 - val_loss: 0.3092 - val_accuracy: 0.8650
    Epoch 24/100
    32/32 [==============================] - 2s 60ms/step - loss: 0.2361 - accuracy: 0.8960 - val_loss: 0.3299 - val_accuracy: 0.8900
    Epoch 25/100
    32/32 [==============================] - 2s 58ms/step - loss: 0.2387 - accuracy: 0.9040 - val_loss: 0.4861 - val_accuracy: 0.8150
    Epoch 26/100
    32/32 [==============================] - 2s 60ms/step - loss: 0.2284 - accuracy: 0.9020 - val_loss: 0.2844 - val_accuracy: 0.8900
    Epoch 27/100
    32/32 [==============================] - 2s 60ms/step - loss: 0.2136 - accuracy: 0.9040 - val_loss: 0.2953 - val_accuracy: 0.8750
    Epoch 28/100
    32/32 [==============================] - 2s 59ms/step - loss: 0.2284 - accuracy: 0.8970 - val_loss: 0.2968 - val_accuracy: 0.8950
    




    <keras.callbacks.History at 0x224f5737308>



# Making a single prediction


```python
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img(
    './dataset/sample/bicycle_or_motorcycle_0000126.png',
    target_size=(32,32)
)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)
print(training_set.class_indices)

if result[0][0] == 0:
    prediction='bicycle'
else:
    prediction='motorcycle'
```

    1/1 [==============================] - 0s 111ms/step
    {'bicycle': 0, 'motorcycle': 1}
    


```python
result
```




    array([[1.]], dtype=float32)




```python
print(prediction)
```

    motorcycle
    


```python
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img(
    './dataset/sample/bicycle_or_motorcycle_000791.png',
    target_size=(32,32)
)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)
print(training_set.class_indices)

if result[0][0] == 0:
    prediction='bicycle'
else:
    prediction='motorcycle'
```

    1/1 [==============================] - 0s 25ms/step
    {'bicycle': 0, 'motorcycle': 1}
    


```python
print(prediction)
```

    bicycle
    


```python

```
