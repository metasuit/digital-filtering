# todo: MAKE THRESHOLD PROPORTIONAL TO BATCH
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy import signal

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)



plt.rcParams['figure.figsize'] = [12, 10] 
plt.rcParams.update({'font.size': 18})

threshold = 100 # Good number for a batchSize of 400, change accordingly

batches = []
avgFFT = []
avgNoisy = []

with open('data/the-oao-data.txt', 'r') as file:
    batches = file.readlines()
maxBatches = len(batches)

dt = 1.0/60
F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
H = np.array([1, 0, 0]).reshape(1, 3)
Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
R = np.array([0.5]).reshape(1, 1)

kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
predictions = []

for batch in batches:
    batchSignal = []
    batch = batch[:-2]
    batch = batch.strip()
    batchSignal = list(map(float, batch.split(',')))
    batchSize = len(batchSignal)

    fhat = np.fft.fft(batchSignal,batchSize)            # Compute the FFFT
    PSD = fhat * np.conj(fhat) / batchSize              # Power spectrum (power per freqeuncy)
    L = np.arange(1,np.floor(batchSize/2), dtype='int') # Only plot the first half of

    ## Filter out ripple noise
    indices = PSD > threshold
    fhat = fhat * indices
    batchSignalFilt = np.fft.ifft(fhat)


    avgFFT.append(np.average([abs(ele) for ele in batchSignalFilt]))
    avgNoisy.append(np.average([abs(ele) for ele in batchSignal]))
    predictions.append(np.dot(H,  kf.predict())[0])

    kf.update(np.average([abs(ele) for ele in batchSignalFilt]))



# Plot results
fig,axs = plt.subplots(1,1)
plt.sca(axs)
plt.plot(np.arange(len(avgNoisy)),avgNoisy,color='r',linewidth=1.5,alpha = 0.5, label='Original Signal')
plt.plot(np.arange(len(avgFFT)),avgFFT,color='b',linewidth=1.5,alpha = 0.5,label='FFT Filtered Signal')
plt.plot(np.arange(len(predictions)),predictions,color='g',linewidth=1.5,label='Kalmann + FFT Filtered Signal')

plt.legend()

plt.show()
