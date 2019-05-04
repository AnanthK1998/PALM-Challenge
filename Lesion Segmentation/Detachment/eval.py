#from unetresnet import *
from precprocess import *

os.chdir("D:/htic/palm/lesion/Detachment/")
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
from segmentation_models.backbones import ResNeXt50,ResNet34
from keras.optimizers import *
from sklearn.metrics import jaccard_similarity_score
#from mAP import mapk
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)
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

model=build_model(pretrained_weights='weights/unetresnext50_Jaccardloss_Adam.49-0.7946-0.9609.hdf5')
imgdir=glob.glob('test/*.jpg')
jac_scores=[]
map_scores=[]
kernel = np.ones((3,3),np.uint8)
for img in tqdm(imgdir):
    imgname=os.path.basename(img).split('.')[0]
    imag=cv2.imread(img)
    shape=imag.shape
    #print(shape)
    imag=cv2.resize(imag,(256,256))
    imag = np.reshape(imag,(1,)+imag.shape)
    result=model.predict(imag)
    pred=cv2.resize(result[0],(shape[1],shape[0]))

    pred=pred*255
    pred[pred<5]=0
    pred[pred>=5]=255
    pred=np.array(pred,dtype=np.uint8)
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('Detachment/'+imgname+'.bmp',pred)
'''
    mask= cv2.imread('val_masks/'+imgname+'.bmp',0)
    #print(pred.shape)
    #print(mask.shape)
    jacc= jaccard_similarity_score(mask,pred,normalize=True)
    pred=pred/255
    mask=mask/255
    #map= mapk(mask,pred,k=10)
    jac_scores.append(jacc)
    #map_scores.append(map)
    
jac_scores=np.array(jac_scores)
#map_scores=np.array(map_scores)
avg1= np.mean(jac_scores)
#avg2=np.mean(map_scores)
print('Average Jaccard Score :')
print(avg1)
#print('\nAverage mAP Score :')
#print(avg2)
'''