"""
2020.04.19

@风不定

基于参数估计方法的nemo鱼颜色分割，此处假设类条件概率密度函数服从正态分布，从nemo模块中调入parametric_estimation类，
分别根据训练集为灰度图和RGB图进行预测

"""

import scipy.io as io
import cv2
from nemo import parametric_estimation

#数据的预处理

#导入训练集数据
data=io.loadmat('F:/nemo_data/array_sample.mat')['array_sample']
img=cv2.imread('F:/nemo_data/nemo.bmp')
mask=io.loadmat('F:/nemo_data/Mask.mat')['Mask']

#得到掩模
mask=mask.reshape(240,320,-1)
img=img*mask

#转化为RGB格式并归一化
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_rgb_standard=img_rgb/255

#转化为灰度图并归一化
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray_standard=img_gray/255

#开始训练数据
model_parametric=parametric_estimation()
model_parametric.train(data)
color_splited_rgb=model_parametric.predict(predict_data=img_rgb_standard,dims=3,display=True)#一维灰度图情况
color_splited_gray=model_parametric.predict(predict_data=img_gray_standard,dims=1,display=True)#三维RGB图情况
