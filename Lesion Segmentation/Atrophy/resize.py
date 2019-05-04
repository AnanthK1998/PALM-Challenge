import glob
import cv2
import os
from tqdm import tqdm
os.chdir('D:/htic/palm/lesion/Atrophy/')
imgdir=glob.glob('train/*.jpg')
for img in tqdm(imgdir):
    imag=cv2.imread(img)
    imgname=os.path.basename(img)
    imag=cv2.resize(imag,(512,512))
    cv2.imwrite('train_resized/'+imgname,imag)
imgdir=glob.glob('masks/*.bmp')
for img in tqdm(imgdir):
    imag=cv2.imread(img)
    imgname=os.path.basename(img)
    imag=cv2.resize(imag,(512,512))
    cv2.imwrite('masks_resized/'+imgname,imag)
imgdir=glob.glob('val/*.jpg')
for img in tqdm(imgdir):
    imag=cv2.imread(img)
    imgname=os.path.basename(img)
    imag=cv2.resize(imag,(512,512))
    cv2.imwrite('val_resized/'+imgname,imag)
imgdir=glob.glob('val_masks/*.bmp')
for img in tqdm(imgdir):
    imag=cv2.imread(img)
    imgname=os.path.basename(img)
    imag=cv2.resize(imag,(512,512))
    cv2.imwrite('val_masks_resized/'+imgname,imag)