import numpy as np
class data_gen:
    def __init__(self, n, x_0):

        self.n=n
        self.x=np.zeros((n,2))
        self.y=np.zeros((n,2))

        self.xi=np.zeros(2)
        self.yi=np.zeros(2)

        self.x[0,:]=x_0[:,0]
        self.u=np.random.normal(0,1,(n,2))
        self.v=np.random.normal(0,1,(n,2))

    def data_1(self):
        for i in range(1,self.n):
            self.x[i,:]=self.state(i,self.x[i-1,:])+ self.u[i,:]
            self.y[i,:]=self.state(i,self.y[i-1,:])+ self.v[i,:]
    def state(self,i,xp):
        self.xi[0]=0.5 * xp[0] - 0.1 * xp[1] + 0.7 * (xp[0] / (1 + xp[0] ** 2)) + 2.2 * np.cos(1.2 * (i - 1))
        self.xi[1]=0.5 * xp[0] - 0.1 * xp[1] + 0.7 * (xp[0] / (1 + xp[0] ** 2)) + 2.2 * np.cos(1.2 * (i - 1))
        return self.xi
    def measurement(self,xi):
        self.yi[0]=(xi[0] * 2) / 9.0 + (xi[1] * 2) / 7.0
        self.yi[1]=(xi[0] * 2) / 6.0 + (xi[1] * 2) / 2.0
        return self.yi
