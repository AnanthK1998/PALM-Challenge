from model import *
import glob
import cv2
import numpy as numpy
from tqdm import tqdm
import os
import csv
os.chdir('D:/htic/palm/classification/')

imgdir=glob.glob('test/*.jpg')
model= ResnetBuilder.build_resnet_50(input_shape=(512,512,3),num_outputs=1,weights='resnet50/resnet50.43-0.0764-1.0000.hdf5')
csvData = [['imgName','Label']]
#PM:0 non-PM:1
#print('Non-PM:\n')
for img in tqdm(imgdir):
    imgname=os.path.basename(img)
    iname=os.path.splitext(imgname)[0]
    imag=cv2.imread(img)
    imag=cv2.resize(imag,(512,512))
    imag=imag/255
    imag = np.reshape(imag,(1,)+imag.shape)
    res= model.predict(imag)
    #print(res[0][0])
    csvData.append([imgname,1.0-res[0][0]])
'''
print('PM:\n')   
imgdir=glob.glob('train/PM/*.jpg')
for img in tqdm(imgdir):
    imgname=os.path.basename(img)
    iname=os.path.splitext(imgname)[0]
    imag=cv2.imread(img)
    imag=cv2.resize(imag,(512,512))
    imag=imag/255
    imag = np.reshape(imag,(1,)+imag.shape)
    res= model.predict(imag)
    print(res[0][0])
    csvData.append([iname,1.0-res[0][0]])

'''

with open('Classification_Results.csv', 'w+') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()