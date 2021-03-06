import os
import cv2
import glob
import numpy as np
np.random.seed(1)
import random as rn 
rn.seed(1)
import tensorflow as tf 
tf.set_random_seed(1)

from tqdm import tqdm
os.chdir("D:/htic/palm/lesion/Atrophy/")
#from focal_loss import *
#from model1 import *
#from data1 import *
#from unetresnet import *

from segmentation_models.unet import Unet
from segmentation_models.backbones import ResNeXt50,InceptionResNetV2
from precprocess import *
from keras.optimizers import *
#from segmentation_models.losses import bce_jaccard_loss
#from segmentation_models.metrics import iou_score
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from dice import *
from jaccard import *
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
  
def build_model(pretrained_weights=None):
    model= Unet('inceptionresnetv2',input_shape=(256,256,3),encoder_weights='imagenet',freeze_encoder=True)
    model.compile(optimizer= Adam(lr = 5e-6),loss=jaccard_distance, metrics=[mean_iou])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

data_gen_args = dict(
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
'''
imgdir=glob.glob('val_masks/*.bmp')
for img in tqdm(imgdir):
    imgname=os.path.basename(img)
    imag=cv2.imread(img)
    imag[imag<255]=0
    cv2.imwrite('D:/htic/PALM-Training400/masks/'+imgname,imag)

'''

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
train_set= traingen.flow_from_directory('',classes=['train'],target_size=(256,256),batch_size=4,class_mode=None,seed=seed)
train_mask_set=traingen.flow_from_directory('',classes=['masks'],target_size=(256,256),batch_size=4,color_mode='grayscale',class_mode=None,seed=seed)
train_generator=zip(train_set,train_mask_set)

val_set=valgen.flow_from_directory('',classes=['val'],target_size=(256,256),batch_size=4,class_mode=None,seed=seed)
val_masks=valgen.flow_from_directory('' ,classes=['val_masks'],target_size=(256,256),batch_size=4,color_mode='grayscale',class_mode=None,seed=seed)
valGene=zip(val_set,val_masks)

#myGene = trainGenerator(4,'','train','masks',data_gen_args,save_to_dir = None)
#valGene= trainGenerator(4,'','val','val_masks',data_gen_args,save_to_dir = None)
#model= unet(pretrained_weights=None)
#model=build_model(pretrained_weights=None)
model= build_model(pretrained_weights='weights/unetinceptionresnetv2_lr-1e-5_Jaccardloss_Adam.02-0.0418-0.9384.hdf5')
model_checkpoint = ModelCheckpoint('weights/unetinceptionresnetv2_lr-5e-6_Jaccardloss_Adam.{epoch:02d}-{val_loss:.4f}-{val_mean_iou:.4f}.hdf5',mode='max',monitor='val_mean_iou',verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_mean_iou',mode='max',factor=0.3, patience=10, min_lr=1e-9, verbose=1)
history=model.fit_generator(train_generator,steps_per_epoch=275,epochs=300,callbacks=[model_checkpoint,reduce_lr],validation_data=valGene,validation_steps= 9,verbose=1,shuffle=False)
