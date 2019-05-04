import glob
import cv2
import os
from tqdm import tqdm
import numpy as np
import glob
import cv2
from tqdm import tqdm
import csv
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from segmentation_models.unet import Unet
from segmentation_models.backbones import ResNeXt50
from keras.optimizers import *
os.chdir('D:/htic/palm/lesion/Atrophy/')
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
    model= Unet('resnext50',input_shape=(256,256,3),encoder_weights='imagenet',freeze_encoder=True)
    model.compile(optimizer= RMSprop(lr = 1e-4),loss='binary_crossentropy', metrics=[mean_iou])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model
imgdir=glob.glob('train/*.jpg')
for img in tqdm(imgdir):
    imgname=os.path.basename(img)
    imag=cv2.imread(img)
    shape=imag.shape
    results=[]
    model=build_model()
    modeldir=glob.glob('unet/*.hdf5')
    for i in tqdm(modeldir):
        modelname=os.path.basename(i)
        model.load_weights('unet/'+modelname)
        imag1=cv2.resize(imag,(256,256))
        imag1 = np.reshape(imag1,(1,)+imag1.shape)
        result=model.predict(imag1)
        pred=result[0]*255
        y=cv2.resize(pred,(shape[0],shape[1]))
        results.append(y)
        del imag1
    results=np.array(results)
    average=np.mean(results,axis=0)
    del results
    
    cv2.imwrite('predicted1/'+imgname,average)
    del average

    