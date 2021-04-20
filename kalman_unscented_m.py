import numpy as np
import scipy
import scipy.linalg
from unsceted_kalman import data_gen

class  UKF:
    def __init__(self, n, m):
        self.n=n
        self.m=m

        self.kapa=0.0
        self.alfa=0.001
        self.beta=2.0
        self.lambda_=(self.n + self.kapa)*self.alfa*self.alfa - self.n
        self.gamma=np.sqrt(self.n+ self.lambda_)
        self.W0m=self.lambda_/(self.n+self.lambda_)
        self.W0c=self.W0m+(1-self.alfa*self.alfa+self.beta)
        self.W=1/(2*(self.n+self.lambda_))
        self.mean=np.zeros((self.n,))
        self.mean_=np.zeros((self.n,))
        self.x=np.zeros((self.n,))
        self.x_p=np.zeros((self.n,))
        self.mea_act=np.zeros((self.m,))
        self.mea=np.zeros((self.m,))
        self.mea_1=np.zeros((self.m,))
        self.mea_2=np.zeros((self.m,))
        self.p_sig=np.zeros((self.n,self.n))
        self.p01=np.zeros((self.n,self.n))
        self.p_=np.zeros((self.n,self.n))
        self.p=np.zeros((self.n,self.n))
        self.sq_p_=np.zeros((self.n,self.n))
        self.sq_p=np.zeros((self.n,self.n))
        self.p_mea=np.zeros((self.m,self.m))
        self.p_mea1=np.zeros((self.m,self.m))
        self.p_cross=np.zeros((self.n,self.m))
        self.P_xy=np.zeros((self.n,self.m))
        self.K=np.zeros((self.n,self.m))
        self.Q=np.zeros((self.n,self.n))
        self.R=np.zeros((self.m,self.m))
        self.S=np.zeros((self.n,self.n))
        
        self.x_sigma = np.zeros((self.n, (2*self.n+1)))
        self.x_sigma_f=np.zeros((self.n,(2*self.n+1)))
        self.mea_sigma=np.zeros((self.m,(2*self.n+1)))
        self.data=0
    def initialize(self,_Q,_R,x_0):
        for i in range(self.n):
            self.Q[i,i]=_Q
        for i in range(self.m):
            self.R[i,i]=_R
        self.mean=x_0[:,0]
        self.mean_=x_0[:,0]
        self.p=self.Q
        self.p_=self.Q
        self.data11=data_gen(self.n,x_0)
    def sigma_pts(self,ini,S):
        self.x_sigma[:,0]=ini
        for k in range(1,self.n+1):
            self.x_sigma[:,k]=ini+self.gamma*self.S[:,k-1]
            self.x_sigma[:,self.n+k]=ini-self.gamma*self.S[:,k-1]
    def nx_st(self,i):
        for j in range(0,2*self.n+1):
            v=self.x_sigma[:,j]
            self.x_sigma_f[:,j]=self.data11.state(i,v)
    def sq_root(self,L):
        sq_root=scipy.linalg.cholesky(L, lower=False)
        return sq_root
    def mea_cal(self):
        for k in range(0,2*self.n+1):
            xi=self.x_sigma_f[:,k]
            self.mea_sigma[:,k]=self.data11.measurement(xi)
            self.mea_1=self.W0m*self.mea_sigma[:,0]
            for k in range(1,2*self.n+1):
                self.mea_1=self.mea_1+self.W*self.mea_sigma[:,k]
    def prediction_update(self,i):
        self.sq_p=self.sq_root(self.p)
        self.sigma_pts(self.mean,self.sq_p)
        self.nx_st(i)
        self.mean_=self.W0m*self.x_sigma_f[:,0]
        for k in range(1,2*self.n+1):
            self.mean_=self.mean_+self.W*self.x_sigma_f[:,k]
        for k in range(0,2*self.n+1):
            self.x_p=self.x_sigma_f[:,k]
            self.x_p=self.x_p-self.mean_
            self.p_sig=np.dot(np.expand_dims(self.x_p,axis=1),np.transpose(np.expand_dims(self.x_p,axis=1)))
            

            if k==0:
                self.p_sig=np.dot(self.W0c,self.p_sig)
            else:
                self.p_sig=np.dot(self.W,self.p_sig)

            self.p_=self.p_+self.p_sig
        self.p_=self.p_+self.Q

        self.sq_p=self.sq_root(self.p_)
        self.sigma_pts(self.mean_,self.sq_p)
        #self.nx_st(i)

        self.mea_cal()
    def measurement_update(self,z):
        for k in range(0,2*self.n+1):
            self.mea=self.mea_sigma[:,k]
            self.mea=self.mea-self.mea_1
            self.p_mea1=np.dot(np.expand_dims(self.mea,axis=1),np.transpose(np.expand_dims(self.mea,axis=1)))
            if k==0:
                self.p_mea1=np.dot(self.W0c,self.p_mea1)
            else:
                self.p_mea1=np.dot(self.W,self.p_mea1)
                self.p_mea=self.p_mea+self.p_mea1
            self.p_mea=self.p_mea+self.R
        for k in range(0,2*self.n+1):
            self.mea=self.mea_sigma[:,k]
            self.mea=self.mea-self.mea_1
            self.x_p=self.x_sigma[:,k]
            self.x_p=self.x_p-self.mean_
            self.p_xy=np.dot(np.expand_dims(self.x_p,axis=1),np.transpose(np.expand_dims(self.mea,axis=1)))
            if k==0:
                self.p_xy=np.dot(self.W0c,self.p_xy)
            else:
                self.p_xy=np.dot(self.W,self.p_xy)
            self.p_cross=self.p_cross+self.p_xy
        self.K=np.dot(self.p_cross,np.linalg.inv(self.p_mea))
        self.y_p=z-self.mea_1
        self.mean=self.mean_+np.dot(self.K,self.y_p)
        self.p=self.p_-np.dot(np.dot(self.K, self.p_mea), np.transpose(self.K))
   
        
        
                
            
        
            

            
       
        
        
            
        
                                
