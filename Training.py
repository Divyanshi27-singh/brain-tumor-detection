import os
import numpy as np
import math
import shutil
import glob
import matplotlib.pyplot as plt
from keras import *
from keras.models import *
from keras.layers import *
import keras
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img , img_to_array
# Use RAW STRING or double backslashes for Windows paths
ROOT_DIR = r"C:\Users\Muskan Singh\Desktop\MACHINE LEARNING\Training"  # Raw string prefix 'r'
# Alternatively: ROOT_DIR = "C:\\Users\\Wuskan Singh\\Desktop\\WACHINE LEARNING\\Training"

number_of_images = {}

# Verify directory exists
if not os.path.exists(ROOT_DIR):
    print(f"ERROR: Directory not found - {ROOT_DIR}")
else:
    # Loop through subdirectories
    for class_dir in os.listdir(ROOT_DIR):
        class_path = os.path.join(ROOT_DIR, class_dir)
        
        if os.path.isdir(class_path):
            num_files = len([
                f for f in os.listdir(class_path) 
                if os.path.isfile(os.path.join(class_path, f))
            ])
            number_of_images[class_dir] = num_files

    # Print results
    print("\nImage Count per Class:")
    for class_name, count in number_of_images.items():
        print(f"{class_name}: {count} images")
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu",input_shape=(224,224,3)))
model.add(Conv2D(filters=36,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dropout(rate=0.25))
model.add(Dense(units=1,activation="sigmoid"))
model.summary()
model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics=['accuracy'])

#preparing our data using data generator
path=(r"C:\Users\Muskan Singh\Desktop\MACHINE LEARNING\Training")

def preprocessingImages(path):
    image_data=ImageDataGenerator(zoom_range=0.2,shear_range=0.2,rescale=1/255,horizontal_flip=True)
    image=image_data.flow_from_directory(directory=path,target_size=(224,224),batch_size=32,class_mode='binary')
    return image
path=(r"C:\Users\Muskan Singh\Desktop\MACHINE LEARNING\Training")
train_data=preprocessingImages(path)

def preprocessingImages(path):
    image_data=ImageDataGenerator(rescale=1/255)
    image=image_data.flow_from_directory(directory=path,target_size=(224,224),batch_size=32,class_mode='binary')
    return image
path=(r"C:\Users\Muskan Singh\Desktop\MACHINE LEARNING\Testing")
test_data=preprocessingImages(path)


# Early stopping and model check point
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es=EarlyStopping(monitor="val_accuracy",min_delta=0.01,patience=6,verbose=1)
mc=ModelCheckpoint(monitor="val_accuracy",filepath="./bestmodel.h5",verbose=1,save_best_only=True)
cd=[es,mc]

hs=model.fit(train_data,steps_per_epoch=8,epochs=30,verbose=1,validation_data=test_data,validation_steps=16,callbacks=cd)
from keras.models import load_model
model=load_model(r"C:\Users\Muskan Singh\Desktop\MACHINE LEARNING\bestmodel.h5")
acc=model.evaluate_generator(test_data)[1]
print(f"The Accuracy of model is {acc*100}%")

path=r"C:\Users\Muskan Singh\Desktop\MACHINE LEARNING\Testing\glioma\Te-gl_0010.jpg"
img=load_img(path,target_size=(224,224))
input_arr=img_to_array(img)/255
input_arr.shape
input_arr=np.expand_dims(input_arr,axis=0)
pred=model.predict_classes(input_arr)[0][0]
pred

if pred==0:
    print("THE IMAGE IS HAVING A BRAIN TUMOR")
else:
    print("THE IMAGE IS NOT HAVING A BRAIN TUMOR")

