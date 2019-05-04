from model import *

os.chdir("D:/htic/palm/classification/")
import numpy as np
import glob
import cv2
from tqdm import tqdm
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Model
'''
fildir= os.listdir("train/")
for fil in fildir:
    alldir=glob.glob("train/"+fil+"/all/*.bmp")
    for img in alldir:
        imag=cv2.imread(img)
        imgname=os.path.basename(img)
        imag=cv2.resize(imag,(225,225))
        cv2.imwrite("train1/"+fil+"/"+imgname,imag)
    hemdir=glob.glob("train/"+fil+"/hem/*.bmp")
    for img in hemdir:
        imag=cv2.imread(img)
        imgname=os.path.basename(img)
        imag=cv2.resize(imag,(225,225))
        cv2.imwrite("train1/"+fil+"/"+imgname,imag)
    


imgdir=glob.glob("train1/fold_0/*.bmp")
x=[]
y=[]

for img in tqdm(imgdir):
    imgname=os.path.basename(img)
    name=imgname.split('.')[0]
    iname=name.split('_')[4]
    if iname=="all":
        y.append(1)
    else:
        y.append(0)
    im=cv2.imread(img)
    x.append(im)
    
y=np.array(y)
x=np.array(x)    

  
metrics= Metrics()  
model = ResnetBuilder.build_resnet_34((225,225,3),1,weights= None)
reduce_lr = ReduceLROnPlateau(monitor='val_acc',factor=0.3, patience=10, min_lr=1e-7, verbose=1)
model_checkpoint = ModelCheckpoint('resnet34/resnet34.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc',verbose=1, save_best_only=True)
history=model.fit(x,y,batch_size=4,epochs=100,verbose=1,validation_split =0.2,callbacks=[model_checkpoint,reduce_lr,metrics])
'''
#metrics=Metrics()
model = ResnetBuilder.build_resnet_50(input_shape=(512,512,3),num_outputs=1,weights='resnet50/resnet50.43-0.0800-1.0000.hdf5',learn=5e-6)
#model= ResNet50(include_top=False,input_shape=(225,225,3),weights="imagenet",classes=2)
#model.compile(optimizer=Adam, loss= 'binary_crossentropy',metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.1, zoom_range = 0.2, horizontal_flip = True,validation_split=0.1)
training_set = train_datagen.flow_from_directory('train',target_size = (512,512),batch_size = 2,class_mode = 'binary',subset='training',seed=1)
validation_set=train_datagen.flow_from_directory('train',target_size = (512,512),batch_size = 2,class_mode = 'binary',subset='validation',seed=1)
print(training_set.class_indices)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.3, patience=5, min_lr=1e-10, verbose=1)
model_checkpoint = ModelCheckpoint('resnet50/resnet50.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
model.fit_generator(training_set,steps_per_epoch = 91,epochs = 300,validation_data=validation_set,validation_steps=10,callbacks=[model_checkpoint])
#print(training_set.class_indices)