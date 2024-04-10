import scipy
import tensorflow as tf
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model #pydot, graphviz

tumor = []
nt= []

directoryyes = 'E:\Computer Vision\Brain Tumor\BrainTumor\yes'
for file in glob.iglob(f'{directoryyes}\\*.jpg'):
    img = cv2.imread(file)      
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
    img = cv2.resize(img, (224, 224)) 
    tumor.append((img, 1)) 

dn = 'E:\\Computer Vision\\Brain Tumor\\BrainTumor\\no'
print(f"Looking for files in: {dn}")  # Debug print to confirm path

for file in glob.iglob(f'{dn}\\*.jpg'):
    print(f"Processing file: {file}")  # Debug print to see which files are being processed
    img = cv2.imread(file)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        nt.append((img, 0))
    else:
        print(f"Failed to read file: {file}")  # Print if a file cannot be read

nt
tumor



all = nt + tumor
data = np.array([item[0] for item in all])
labels = np.array([item[1] for item in all])
labels
plt.imshow(data[20])
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, shuffle=True, test_size=0.2, random_state=3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,shuffle=True, test_size=0.2, random_state=3)

#X_train = X_train / 255.0
##X_val = X_val / 255.0
##X_test = X_test / 255.0


#Data Augmentation 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_augmentation = ImageDataGenerator(
    rescale=1./255, # Rescaling factor (applied before any other transformation)
    rotation_range=40, # Degree range for random rotations
    #width_shift_range=0.2, # Range (as a fraction of total width) for random horizontal shifts
    #height_shift_range=0.2, # Range (as a fraction of total height) for random vertical shifts
    #shear_range=0.2, # Shear intensity (shear angle in counter-clockwise direction)
    zoom_range=0.2, # Range for random zoom
    horizontal_flip=True, # Randomly flip inputs horizontally
    fill_mode='wrap' # Strategy for filling in newly created pixels
)

# Assuming X_train and y_train are your training data and labels, respectively
train_generator = data_augmentation.flow(X_train, y_train, batch_size=4)




    

batch_size = 5
img_height = 180
img_width = 180
data[1].shape
labels.shape
#train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory,validation_split=.2,subset='training',seed=123,image_size=(img_height,img_width),batch_size=batch_size)
v#al_ds = tf.keras.preprocessing.image_dataset_from_directory(directory,validation_split=.2,subset='validation',seed=123,image_size=(img_height,img_width),batch_size=batch_size)

#AUTOTUNE = tf.data.AUTOTUNE
#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#num_classes = 2

#model = tf.keras.Sequential([
 # tf.keras.layers.Rescaling(1./255),
  #tf.keras.layers.Conv2D(32, kernel_size = (3,3), padding='same' ,activation='relu',input_shape=(224,224,3)),
  #tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
  #tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
  ##tf.keras.layers.MaxPooling2D(),
  #tf.keras.layers.Flatten(),
  #tf.keras.layers.Dense(64),
  #tf.keras.layers.Dense(1,activation='softmax')
#])
from tensorflow import keras

model = keras.Sequential([

  keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), kernel_initializer='glorot_uniform'),
  keras.layers.MaxPooling2D((2, 2)),
  keras.layers.Conv2D(64, (3, 3), activation='leaky_relu'),
  keras.layers.MaxPooling2D((2, 2)),
  keras.layers.Conv2D(128, (3, 3), activation='relu'),
  keras.layers.MaxPooling2D((2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['Accuracy','Recall'])

epochs=10
history = model.fit(
  train_generator,
  validation_data=[X_val,y_val],
  epochs=epochs,
  batch_size= 4
)
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import graphviz
import pydot

plot_model(model, to_file='model_architecture.png', show_shapes=True)