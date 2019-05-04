from model1 import *
from data1 import *
from keras.optimizers import *
#from segmentation_models.losses import bce_jaccard_loss
#from segmentation_models.metrics import iou_score
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import os
os.chdir('D:/htic/palm/disc/')
traingen=ImageDataGenerator(rescale=1./255,
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
valgen=ImageDataGenerator(rescale=1./255,
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
seed=1
train_set= traingen.flow_from_directory('',classes=['train'],target_size=(256,256),batch_size=4,class_mode=None,color_mode='grayscale',seed=seed)
train_mask_set=traingen.flow_from_directory('',classes=['masks'],target_size=(256,256),batch_size=4,color_mode='grayscale',class_mode=None,seed=seed)
train_generator=zip(train_set,train_mask_set)

val_set=valgen.flow_from_directory('',classes=['val'],target_size=(256,256),batch_size=4,class_mode=None,color_mode='grayscale',seed=seed)
val_masks=valgen.flow_from_directory('' ,classes=['val_masks'],target_size=(256,256),batch_size=4,color_mode='grayscale',class_mode=None,seed=seed)
valGene=zip(val_set,val_masks)

model=unet(pretrained_weights=None)
model_checkpoint = ModelCheckpoint('unet/unet_lr=1e-4_BCE_Adam.{epoch:02d}-{val_loss:.4f}-{val_mean_iou:.4f}.hdf5',mode='max',monitor='val_mean_iou',verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_mean_iou',mode='max',factor=0.3, patience=10, min_lr=1e-9, verbose=1)
history=model.fit_generator(train_generator,steps_per_epoch=491,epochs=100,callbacks=[model_checkpoint,reduce_lr],validation_data=valGene,validation_steps= 5,verbose=1)
