"""
2020.04.19

@风不定

nemo鱼颜色分割的三个训练与预测模型，包括参数估计的一个类parametric_estimation和非参数估计的两个类Bayes_classification_gray/Bayes_classification_rgb

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 
from math import*
from numpy.linalg import*


#非参数灰度图模型
class Bayes_classification_gray:
    def __init__(self):
        pass
    
    def train(self,data):
        data_pd=pd.DataFrame(np.array(data))
        self.data=data_pd
        self.data_positive=data_pd[:][data_pd[4]==1]
        self.data_negative=data_pd[:][data_pd[4]==-1]
        self.N_positive=len(self.data_positive)
        self.N_negative=len(self.data_negative)
        self.p_positive=len(self.data_positive)/len(self.data)
        self.p_negative=1-self.p_positive
        self.data_positive_intensity=np.array(self.data_positive[0])
        self.data_negative_intensity=np.array(self.data_negative[0])
        

    #高斯核
    def Gaussian_kernel(self,x,xi,rou=1):
        k=exp(-(x-xi)**2/(2*rou**2))/(sqrt(2*pi)*rou)
        return k
    
    #立方体核
    def cube_kernel(self,x,xi,h=0.2):
        if abs(x-xi)>h/2:
            return 0
        else:
            return 1/h

    #直方图法    
    def histogram(self,bin_width=0.05,draw=True):
        bin_number=int(1/bin_width)
        bins_positive=np.zeros(bin_number)
        bins_negative=bins_positive.copy()
        p_hat_positive=np.zeros(bin_number)
        p_hat_negative=p_hat_positive.copy()
        
        for i in range(len(self.data_positive_intensity)):
            index=int(self.data_positive_intensity[i]/bin_width)
            bins_positive[index]+=1
        p_hat_positive=bins_positive/(len(self.data_positive_intensity)*bin_width)
            
        for i in range(len(self.data_negative_intensity)):
            index=int(self.data_negative_intensity[i]/bin_width)
            bins_negative[index]+=1
        p_hat_negative=bins_negative/(len(self.data_negative_intensity)*bin_width)


        #绘制pdf的函数图像
        if draw:
            fig=plt.figure()
            sns.set()
            ax1=fig.add_subplot(1,2,1)
            ax1=sns.distplot(p_hat_positive,bins=bin_number,rug_kws=True)
            ax1.set_title('+1 pdf,histogram')
            ax1.set_ylabel('pdf')
            plt.show()
            sns.set()
            ax2=fig.add_subplot(1,2,2)
            ax2.set_title('-1 pdf,histogram')
            ax2=sns.distplot(p_hat_negative,bins=bin_number,rug_kws=True)
            ax2.set_ylabel('pdf')
            plt.show()
            
        self.p_positive_histogram=p_hat_positive
        self.p_negative_histogram=p_hat_negative

    #kn近邻法
    def kNN(self,kn=10,draw=True):
       
        positive_sorted=sorted(self.data_positive_intensity)
        negative_sorted=sorted(self.data_negative_intensity)
        dis_positive={i:[] for i in np.arange(0.00,1.01,0.01).round(2)}
        pdf_positive=dis_positive.copy()
        dis_negative={i:[] for i in np.arange(0.00,1.01,0.01).round(2)}
        pdf_negative=dis_negative.copy()
        meshgrid=np.arange(0,1.01,0.01).round(2)
        
        for value in meshgrid:
            for pixel in positive_sorted:
                distance=abs(value-pixel)
                dis_positive[value].append(distance)
            dis_positive[value]=sorted(dis_positive[value])[:kn]
            pdf_positive[value]=kn/(self.N_positive*2*dis_positive[value][-1])
            
        for value in meshgrid:
            for pixel in negative_sorted:
                distance=abs(value-pixel)
                dis_negative[value].append(distance)
            dis_negative[value]=sorted(dis_negative[value])[:kn]
            pdf_negative[value]=kn/(self.N_negative*2*dis_negative[value][-1]) 
            
        if draw:
            fig=plt.figure()
            dic_positive={'pdf':list(pdf_positive.values()),'intensity':np.arange(0,1.01,0.01).round(2)}
            dic_negative={'pdf':list(pdf_negative.values()),'intensity':np.arange(0,1.01,0.01).round(2)}
            sns.set()
            ax1=fig.add_subplot(1,2,1)
            ax1=sns.lineplot(x='intensity',y='pdf',data=pd.DataFrame(dic_positive))
            ax1.set_title('+1 pdf,kNN,kn=%d'%kn)
            ax1.set_ylabel('pdf')
            plt.show()
            sns.set()
            ax2=fig.add_subplot(1,2,2)
            ax2.set_title('-1 pdf,kNN,kn=%d'%kn)
            ax2=sns.lineplot(x='intensity',y='pdf',data=pd.DataFrame(dic_negative))
            ax2.set_ylabel('pdf')
            plt.show()
            
            
        self.pdf_positive_kNN=pdf_positive
        self.pdf_negative_kNN=pdf_negative

    #parzen窗法
    def parzen(self,rou=0.10,h=0.2,kernel='Gauss',draw=True):
        
        sum=0
        intensities=np.arange(0.00,1.01,0.01).round(2)
        pdf_positive={i:[] for i in intensities}
        pdf_negative={i:[] for i in intensities}
        for intensity in intensities:
            for pixel in self.data_positive_intensity:
                if kernel=='Gauss':
                    sum+=self.Gaussian_kernel(intensity,pixel,rou)
                else:
                    sum+=self.cube_kernel(intensity,pixel,h)      
            pdf_positive[intensity]=1/self.N_positive*sum
            sum=0
            
        for intensity in intensities:
            for pixel in self.data_negative_intensity:
                if kernel=='Gauss':
                    sum+=self.Gaussian_kernel(intensity,pixel,rou)
                else:
                    sum+=self.cube_kernel(intensity,pixel,h)      
            pdf_negative[intensity]=1/self.N_negative*sum
            sum=0
            
            
        if draw:
            fig=plt.figure()
            dic_positive={'pdf':list(pdf_positive.values()),'intensity':np.arange(0,1.01,0.01).round(2)}
            dic_negative={'pdf':list(pdf_negative.values()),'intensity':np.arange(0,1.01,0.01).round(2)}
            sns.set()
            ax1=fig.add_subplot(1,2,1)
            ax1=sns.lineplot(x='intensity',y='pdf',data=pd.DataFrame(dic_positive))
            ax1.set_title('+1 pdf,parzen,kernel=%s'%kernel)
            ax1.set_ylabel('pdf')
            plt.show()
            sns.set()
            ax2=fig.add_subplot(1,2,2)
            ax2.set_title('-1 pdf,parzen,kernel=%s'%kernel)
            ax2=sns.lineplot(x='intensity',y='pdf',data=pd.DataFrame(dic_negative))
            ax2.set_ylabel('pdf')
            plt.show()
            

        self.pdf_positive_parzen=pdf_positive
        self.pdf_negative_parzen=pdf_negative

    def classify(self,intensity,method,bin_width=0.05,kn=10,rou=0.10,h=0.2,kernel='Gauss'):
        
        
        if method=='histogram':
            index=int(intensity/bin_width)
            classification=(self.p_positive_histogram[index]*self.p_positive>self.p_negative_histogram[index]*self.p_negative)
                
        if method=='kNN':
            index=round(intensity,2)
            classification=(self.pdf_positive_kNN[index]*self.p_positive>self.pdf_negative_kNN[index]*self.p_negative)
                
        if method=='parzen':
            index=round(intensity,2)
            classification=(self.pdf_positive_parzen[index]*self.p_positive>self.pdf_negative_parzen[index]*self.p_negative)
            
                
        return classification        
        
    
    def predict(self,predict_data,method,kernel='Gauss',bin_width=0.05,kn=10,rou=0.10,h=0.2,display=True,draw=True):
        shape=predict_data.shape
        new_img=np.zeros((shape[0],shape[1],3))
        
        if method=='histogram':
            self.histogram(bin_width,draw)
        if method=='kNN':
            self.kNN(kn,draw)
        if method=='parzen':
            self.parzen(rou,h,kernel,draw)
            
        for i in range(shape[0]):
            for j in range(shape[1]):
                if predict_data[i][j]!=0 :
                    if self.classify(predict_data[i][j],method,bin_width,kn,rou,h,kernel):
                        new_img[i][j]=np.array([0,0,255])
                    else:
                        new_img[i][j]=np.array([255,0,0])
        
        if display:
            if method=='histogram':
                cv2.imshow('color_splited_intensity,histogram',new_img)
            if method=='kNN':
                cv2.imshow('color_splited_intensity,kNN,kn=%d'%kn,new_img)
            if method=='parzen':
                if kernel=='Gauss':
                    cv2.imshow('color_splited_intensity,Guassian_parzen',new_img)
                else:
                    cv2.imshow('color_splited_intensity,cube_parzen',new_img)
                
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return new_img



class Bayes_classification_rgb:
    def __init__(self):
        pass
    
    def train(self,data):
        data_pd=pd.DataFrame(np.array(data))
        self.data=data_pd
        self.data_positive=data_pd[:][data_pd[4]==1]
        self.data_negative=data_pd[:][data_pd[4]==-1]
        self.N_positive=len(self.data_positive)
        self.N_negative=len(self.data_negative)
        self.p_positive=len(self.data_positive)/len(self.data)
        self.p_negative=1-self.p_positive
        self.data_positive_rgb=np.array(self.data_positive.iloc[:,1:4])
        self.data_negative_rgb=np.array(self.data_negative.iloc[:,1:4])
    
    def Gaussian_kernel(self,x,xi,rou=0.1,Q=np.eye(3)):
        k=np.exp(-(x-xi).dot(inv(Q)).dot(x-xi)/(2*rou**2))/(sqrt((2*pi)**3*rou**6*det(Q)))
        return k

    def cube_kernel(self,x,xi,h=0.2):
        for item in zip(x,xi):
            if abs(item[0]-item[1])>h/2:
                return 0
        else:
            return 1/h**3
        
    def hypersphere(self,x,xi,r=0.2):
        if norm(x-xi)>r:
            return 0
        else:
            return 4/3*pi*r**3
        
    def histogram(self,bin_width=0.02):
        
        bin_number=int(1/bin_width)
        meshgrid=np.arange(0,1.01,bin_width).round(2)
        pdf_positive={(r,g,b):0 for r in meshgrid for g in meshgrid for b in meshgrid}
        pdf_negative={(r,g,b):0 for r in meshgrid for g in meshgrid for b in meshgrid}
        
        for i in range(self.N_positive):
            index=self.data_positive_rgb[i]/bin_width
            key=(round(int(index[0])*bin_width,2),round(int(index[1])*bin_width,2),round(int(index[2])*bin_width,2))
            pdf_positive[key]+=1
        for key in pdf_positive:
            pdf_positive[key]=pdf_positive[key]/(self.N_positive*bin_width**3)
            
        for i in range(self.N_negative):
            index=self.data_negative_rgb[i]/bin_width
            key=(round(int(index[0])*bin_width,2),round(int(index[1])*bin_width,2),round(int(index[2])*bin_width,2))
            pdf_negative[key]+=1
        for key in pdf_negative:
            pdf_negative[key]=pdf_negative[key]/(self.N_negative*bin_width**3)
            
        self.p_positive_histogram=pdf_positive
        self.p_negative_histogram=pdf_negative
    
    def kNN(self,kn=10,step=0.2):
        
        meshgrid=np.arange(0,1.01,step).round(2)
        dis_positive={(r,g,b):[] for r in meshgrid for g in meshgrid for b in meshgrid}
        pdf_positive=dis_positive.copy()
        dis_negative={(r,g,b):[] for r in meshgrid for g in meshgrid for b in meshgrid}
        pdf_negative=dis_negative.copy()
        
        for rgb in dis_positive:
            for pixel in self.data_positive_rgb:
                distance=norm(list(rgb)-pixel)
                dis_positive[rgb].append(distance)
            
            dis_positive[rgb]=sorted(dis_positive[rgb])[:kn]
            pdf_positive[rgb]=kn/(self.N_positive*8*dis_positive[rgb][-1]**3)
            
        for rgb in dis_negative:
            for pixel in self.data_negative_rgb:
                distance=norm(list(rgb)-pixel)
                dis_negative[rgb].append(distance)
            
            dis_negative[rgb]=sorted(dis_negative[rgb])[:kn]
            pdf_negative[rgb]=kn/(self.N_negative*8*dis_negative[rgb][-1]**3)
            
        self.pdf_positive_kNN=pdf_positive
        self.pdf_negative_kNN=pdf_negative
            
    def parzen(self,rou=1,Q=np.eye(3),h=0.2,r=0.2,kernel='Gauss',step=0.2):
        
        sum=0
        meshgrid=np.arange(0,1.01,step).round(2)
        pdf_positive={(r,g,b):0 for r in meshgrid for g in meshgrid for b in meshgrid}
        pdf_negative={(r,g,b):0 for r in meshgrid for g in meshgrid for b in meshgrid}
        
        for rgb in pdf_positive:
            for pixel in self.data_positive_rgb:
                if kernel=='Gauss':
                    sum+=self.Gaussian_kernel(list(rgb),pixel,rou,Q)
                elif kernel=='cube':
                    sum+=self.cube_kernel(list(rgb),pixel,h)
                else:
                    sum+=self.hypersphere(list(rgb),pixel,r)
                
            pdf_positive[rgb]=1/self.N_positive*sum
            sum=0
            
        for rgb in pdf_negative:
            for pixel in self.data_negative_rgb:
                if kernel=='Gauss':
                    sum+=self.Gaussian_kernel(list(rgb),pixel,rou,Q)
                elif kernel=='cube':
                    sum+=self.cube_kernel(list(rgb),pixel,h)
                else:
                    sum+=self.hypersphere(list(rgb),pixel,r)
                
            pdf_negative[rgb]=1/self.N_negative*sum
            sum=0
            
        self.pdf_positive_parzen=pdf_positive
        self.pdf_negative_parzen=pdf_negative

    
    def classify(self,rgb,method,bin_width=0.05,kn=10,rou=0.10,Q=np.eye(3),h=0.2,step=0.2,r=0.2,kernel='Gauss'):
        
        
        if method=='histogram':
            index=rgb/bin_width
            key=(round(int(index[0])*bin_width,2),round(int(index[1])*bin_width,2),round(int(index[2])*bin_width,2))
            classification=(self.p_positive_histogram[key]*self.p_positive>self.p_negative_histogram[key]*self.p_negative)
                
        if method=='kNN':
            index=rgb/step
            key=(round(int(index[0])*step,2),round(int(index[1])*step,2),round(int(index[2])*step,2))
            classification=(self.pdf_positive_kNN[key]*self.p_positive>self.pdf_negative_kNN[key]*self.p_negative)
                
        if method=='parzen':
            index=rgb/step
            key=(round(int(index[0])*step,2),round(int(index[1])*step,2),round(int(index[2])*step,2))
            classification=(self.pdf_positive_parzen[key]*self.p_positive>self.pdf_negative_parzen[key]*self.p_negative)
              
        return classification   
    
    
    def predict(self,predict_data,method,kernel='Gauss',bin_width=0.05,kn=10,rou=0.10,Q=np.eye(3),h=0.2,step=0.2,r=0.2,display=True):
        shape=predict_data.shape
        new_img=np.zeros((shape[0],shape[1],3))
        
        if method=='histogram':
            self.histogram(bin_width)
        if method=='kNN':
            self.kNN(kn,step)
        if method=='parzen':
            self.parzen(rou,Q,h,r,kernel,step)
            
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (predict_data[i][j]!=0).any() :
                    if self.classify(predict_data[i][j],method,bin_width,kn,rou,Q,h,step,r,kernel):
                        new_img[i][j]=np.array([0,0,255])
                    else:
                        new_img[i][j]=np.array([255,0,0])
        
        if display:
            if method=='histogram':
                cv2.imshow('color_splited_rgb,histogram',new_img)
            if method=='kNN':
                cv2.imshow('color_splited_rgb,kNN,kn=%d'%kn,new_img)
            if method=='parzen':
                if kernel=='Gauss':
                    cv2.imshow('color_splited_rgb,Guassian_parzen',new_img)
                elif kernel=='cube':
                    cv2.imshow('color_splited_rgb,cube_parzen',new_img)
                else:
                    cv2.imshow('color_splited_rgb,hypersphere_parzen',new_img)
                
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return new_img



class parametric_estimation:
    
    def __init__(self):
        pass
    
    
    def train(self,data):
        data_pd=pd.DataFrame(np.array(data))
        self.data=data_pd
        self.data_positive=data_pd[:][data_pd[4]==1]
        self.data_negative=data_pd[:][data_pd[4]==-1]
        self.N_positive=len(self.data_positive)
        self.N_negative=len(self.data_negative)
        self.p_positive=len(self.data_positive)/len(self.data)
        self.p_negative=1-self.p_positive
        self.data_positive_intensity=np.array(self.data_positive[0])
        self.data_negative_intensity=np.array(self.data_negative[0])
        self.data_positive_rgb=np.array(self.data_positive.iloc[:,1:4])
        self.data_negative_rgb=np.array(self.data_negative.iloc[:,1:4])
        mean_dict,sigma_dict=self.parametric_calculate()
        self.Wp_1=-1/(2*sigma_dict['spi'])
        self.Wn_1=-1/(2*sigma_dict['sni'])
        self.wp_1=mean_dict['mpi']/sigma_dict['spi']
        self.wn_1=mean_dict['mni']/sigma_dict['sni']
        self.wp0_1=-1/(2*sigma_dict['spi'])*mean_dict['mpi']**2-1/2*log(abs(sigma_dict['spi']))+log(self.p_positive)
        self.wn0_1=-1/(2*sigma_dict['sni'])*mean_dict['mni']**2-1/2*log(abs(sigma_dict['sni']))+log(self.p_negative)
        self.Wp_3=-1/2*inv(sigma_dict['sprgb'])
        self.Wn_3=-1/2*inv(sigma_dict['snrgb'])
        self.wp_3=np.dot(inv(sigma_dict['sprgb']),mean_dict['mprgb'])
        self.wn_3=np.dot(inv(sigma_dict['snrgb']),mean_dict['mnrgb'])
        self.wp0_3=-1/2*mean_dict['mprgb'].dot(inv(sigma_dict['sprgb'])).dot(mean_dict['mprgb'])
        -1/2*log(det(sigma_dict['sprgb']))+log(self.p_positive)
        self.wn0_3=-1/2*mean_dict['mnrgb'].dot(inv(sigma_dict['snrgb'])).dot(mean_dict['mnrgb'])
        -1/2*log(det(sigma_dict['snrgb']))+log(self.p_negative)
        
        
        
    def parametric_calculate(self):
        mean_positive_intensity=1/self.N_positive*np.sum(self.data_positive_intensity,axis=0)
        mean_negative_intensity=1/self.N_negative*np.sum(self.data_negative_intensity,axis=0)
        mean_positive_rgb=1/self.N_positive*np.sum(self.data_positive_rgb,axis=0)
        mean_negative_rgb=1/self.N_negative*np.sum(self.data_negative_rgb,axis=0)
        sigma_positive_intensity=1/(self.N_positive-1)*np.sum((self.data_positive_intensity-mean_positive_intensity)**2)                                                 
        sigma_negative_intensity=1/(self.N_negative-1)*np.sum((self.data_negative_intensity-mean_negative_intensity)**2)
        sigma_positive_rgb=1/(self.N_positive-1)*np.sum([np.outer(i,i) for i in self.data_positive_rgb-mean_positive_rgb],axis=0)
        sigma_negative_rgb=1/(self.N_negative-1)*np.sum([np.outer(i,i) for i in self.data_negative_rgb-mean_negative_rgb],axis=0)
        mean_dict={'mpi':mean_positive_intensity,'mni':mean_negative_intensity,'mprgb':mean_positive_rgb,'mnrgb':mean_negative_rgb}
        sigma_dict={'spi':sigma_positive_intensity,'sni':sigma_negative_intensity,
                   'sprgb':sigma_positive_rgb,'snrgb':sigma_negative_rgb}
        
        return mean_dict,sigma_dict
    
    
    def hyperplane(self,x,dims):
        if dims==1:
            hyperplane=(self.Wp_1-self.Wn_1)*x**2+(self.wp_1-self.wn_1)*(x)+self.wp0_1-self.wn0_1
        else:
            hyperplane=x.dot(self.Wp_3-self.Wn_3).dot(x)+(self.wp_3-self.wn_3).dot(x)+self.wp0_3-self.wn0_3
        #print(np.sign(hyperplane))    
        return np.sign(hyperplane)
            
    
    def predict(self,predict_data,dims,display=True):
        shape=predict_data.shape
        new_img=np.zeros((shape[0],shape[1],3))
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (predict_data[i][j]!=0 if dims==1 else (predict_data[i][j]!=0).any()):
                    if self.hyperplane(predict_data[i][j],dims)==1:
                        new_img[i][j]=np.array([0,0,255])
                    else:
                        new_img[i][j]=np.array([255,0,0])
        
        if display:
            if dims==1:
                cv2.imshow('color_splited_intensity',new_img)
            else:
                cv2.imshow('color_splited_rgb',new_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return new_img
