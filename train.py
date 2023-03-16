%matplotlib inline
import PIL
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
#from torchvision import datasets
from skimage.filters import gabor
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
import glob
import cv2
import os
from PIL import Image

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

# Capture images and labels into arrays.
# Start by creating empty lists.
# SIZE = 128
train_images = []
train_images_topleft = []
train_images_topRight = []
train_images_bottomLeft = []
train_images_bottomRight = []
train_labels = []
final_images = []
# for directory_path in glob.glob("cell_images/train/*"):
for directory_path in glob.glob(r"C:/Users/fsshi/jupyter notebooks/train2/*"):
    label = directory_path.split("\\")[-1]
    # print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(os.path.basename(img_path))
        imm = cv2.imread(img_path, 1)  # Reading color images
        im = cv2.filter2D(src=imm, ddepth=-1, kernel=kernel)
        print(im.shape)
        h, w = im.shape[:2]
        print(h)
        print(w)
        centerX, centerY = (w // 2), (h // 2)

        topLeft = im[0:centerY, 0:centerX]
        topRight = im[0:centerY, centerX:w]
        bottomLeft = im[centerY:h, 0:centerX]
        bottomRight = im[centerY:h, centerX:w]

        train_images_topleft.append(topLeft)
        train_images_topRight.append(topRight)
        train_images_bottomLeft.append(bottomLeft)
        train_images_bottomRight.append(bottomRight)
        final_images = train_images_topleft + train_images_topRight + train_images_bottomLeft + train_images_bottomRight

        # img = cv2.resize(img, (SIZE, SIZE)) #Resize images
        train_images.append(im)
        train_labels.append(label)

import pandas as pd
featLength = 2+5+2
trainFeats = np.zeros((len(train_images_bottomRight),featLength)) #Feature vector of each image is of size 1x1030
for tr in tqdm.tqdm_notebook(range(len(train_images_bottomRight))):

    img = train_images_bottomRight[tr][0] #One image at a time
    #print(img.shape)
    img = Image.fromarray(img)
    img_gray = img.convert('L') #Converting to grayscale
    img_arr = np.array(img.getdata()).reshape(img.size[1],img.size[0]) #Converting to array
    # LBP
    feat_lbp = local_binary_pattern(img_arr,5,2,'uniform').reshape(img.size[0]*img.size[1])
    lbp_hist,_ = np.histogram(feat_lbp,8)
    lbp_hist = np.array(lbp_hist,dtype=float)
    lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
    lbp_energy = np.nansum(lbp_prob**2)
    lbp_entropy = -np.nansum(np.multiply(lbp_prob,np.log2(lbp_prob)))
    # GLCM
    gCoMat = greycomatrix(img_arr, [2], [0],256,symmetric=True, normed=True)
    contrast = greycoprops(gCoMat, prop='contrast')
    dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
    homogeneity = greycoprops(gCoMat, prop='homogeneity')
    energy = greycoprops(gCoMat, prop='energy')
    correlation = greycoprops(gCoMat, prop='correlation')
    feat_glcm = np.array([contrast[0][0],dissimilarity[0][0],homogeneity[0][0],energy[0][0],correlation[0][0]])
    # Gabor filter
    gaborFilt_real,gaborFilt_imag = gabor(img_arr,frequency=0.6)
    gaborFilt = (gaborFilt_real**2+gaborFilt_imag**2)//2
    gabor_hist,_ = np.histogram(gaborFilt,8)
    gabor_hist = np.array(gabor_hist,dtype=float)
    gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
    gabor_energy = np.nansum(gabor_prob**2)
    gabor_entropy = -np.nansum(np.multiply(gabor_prob,np.log2(gabor_prob)))
    # Concatenating features(2+5+2)
    concat_feat = np.concatenate(([lbp_energy,lbp_entropy],feat_glcm,[gabor_energy,gabor_entropy]),axis=0)
    trainFeats[tr,:] = concat_feat #Stacking features vectors for each image
    df4 = pd.DataFrame({'lbp_energy': trainFeats[:,0],
                       'lbp_entropy': trainFeats[:,1],
                       #'contrast': trainFeats[:,2],
                       'dissimilarity': trainFeats[:,3],
                       'homogeneity': trainFeats[:,4],
                       'energy': trainFeats[:,5],
                       'correlation': trainFeats[:,6],
                       'gabor_energy': trainFeats[:,7],
                       'gabor_entropy': trainFeats[:,8]})


df4 = (df4-df4.min())/(df4.max()-df4.min())
train_labels = np.array(train_labels)


from sklearn.naive_bayes import GaussianNB

# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, y_train)
yhat_prob = model.predict_proba(X_test)
print('Predicted Probabilities: ', yhat_prob)
# Predict Output
predicted = model.predict(X_test)

print("Actual Value:", y_test)
print("Predicted Value:", predicted)


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

y_pred = model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)
print(classification_report(y_pred, y_test))

import pickle
pickle.dump(model, open('model.pkl', 'wb'))
