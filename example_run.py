import os.path
from unsceted_kalman import data_gen
from kalman_unscented_m import UKF
import numpy as np
from matplotlib import pyplot as plt

def estimateState():
    n=2
    m=2
    x_0=np.zeros((n,1))
    x_0[0, 0] = 0.5
    x_0[1, 0] = 0.5

    data11=data_gen(500,x_0)
    data11.data_1()

    ukf=UKF(n,m)
    dataX=data11.x
    dataY=data11.y

    size=dataX.shape[0]
    ukf.initialize(0.1,0.1,x_0)

    prediction=0
    measurement=np.zeros((m,1))
    for i in range(size):
        prediction=i
        measurement=dataY[i,:]
        ukf.prediction_update(prediction)
        ukf.measurement_update(measurement)
        #print(ukf.mean)
        #print(dataX)
        plt.plot(ukf.mean)
        plt.plot(dataX)
        plt.show()
estimateState()

    
    
    
