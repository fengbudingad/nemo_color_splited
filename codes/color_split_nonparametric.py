"""
2020.04.19

@风不定

基于非参数估计方法的nemo鱼颜色分割，从nemo模块中调入Bayes_classification_gray类和Bayes_classification_rgb类，
分别根据训练集为灰度图和RGB图进行预测
"""

import scipy.io as io
import cv2
from nemo import Bayes_classification_gray,Bayes_classification_rgb

#读入训练集数据
data=io.loadmat('F:/nemo_data/array_sample.mat')['array_sample']
img=cv2.imread('F:/nemo_data/nemo.bmp')
mask=io.loadmat('F:/nemo_data/Mask.mat')['Mask']

#获得掩模图像
mask=mask.reshape(240,320,-1)
img=img*mask

#转化为RGB格式并归一化
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_rgb_standard=img_rgb/255

#转化为灰度图并归一化
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('original rgb image',img)
img_gray_standard=img_gray/255

#分别调用用于灰度图和RGB图的两个类
model_intensity=Bayes_classification_gray()
model_intensity.train(data)
ans=model_intensity.predict(predict_data=img_gray_standard,method='histogram',draw=True)#使用直方图方法
model_rgb=Bayes_classification_rgb()
model_rgb.train(data)
ans=model_rgb.predict(predict_data=img_rgb_standard,method='kNN',kn=10)#使用kn近邻法
