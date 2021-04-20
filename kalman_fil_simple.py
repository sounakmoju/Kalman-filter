from numpy import transpose
from numpy import dot
from numpy import dot, sum, linalg
from numpy.linalg import inv
from numpy import eye
from numpy import zeros
import numpy as np
import random
import matplotlib.pyplot as plt
def predict(X,P,A,Q,B,U):
    X=dot(A,X)+ dot(B,U)
    P=dot(A,dot(P,transpose(A)))+ Q
    
    return(X,P) 
def update(X,P,Y,H,R):
    i_mea=(dot(H,X))
    IS=R+dot(H,dot(P,transpose(H)))
    K=dot(P,dot(transpose(H),inv(IS)))
    X=X+dot(K,(Y-(i_mea)))
    P=P-dot(K,dot(IS,transpose(K)))
    k1=dot(K,(Y-(i_mea)))
    #print(isinstance(X,list))
    #print(i_mea)
    #print(X,P,K,i_mea,IS)
    return(X,P,K,i_mea,IS)
from numpy import *
from numpy.linalg import inv

dt = .1

X = array([[0.0], [0.0], [0.1], [0.1]])
P = diag((0.01, 0.01, 0.01, 0.01))
A = array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0,\
1]])
#print((B.shape))
Q = eye(X.shape[0])
B = eye(X.shape[0])
U = zeros((X.shape[0],1))
Y = array([[X[0,0] + abs(np.random.randn(1)[0])], [X[1,0] +\
abs(np.random.randn(1)[0])],[X[2,0] + abs(np.random.randn(1)[0])],[X[3,0] + abs(np.random.randn(1)[0])]])
#print(isinstance(Y, list))
H = array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
R = eye(Y.shape[0])
print(R.shape)

N_iter = 50

for i in arange(0, N_iter):
    (X, P) = predict(X, P, A, Q, B, U)
    (X, P, K, i_mea, IS)=(update(X, P, Y, H, R))
    Y1 = array([[X[0,0] + abs(np.random.randn(1)[0])], [X[1,0] +\
abs(np.random.randn(1)[0])],[X[2,0] + abs(0.1*np.random.randn(1)[0])],[X[3,0] + abs(0.1*np.random.randn(1)[0])]])
    t=[]
    t.append(i)
print(Y1.shape)
plt.plot(Y1)
plt.plot(X[:,1)
plt.show()
          
