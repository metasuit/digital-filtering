# todo: MAKE THRESHOLD PROPORTIONAL TO BATCH
import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = [12, 10] 
plt.rcParams.update({'font.size': 8})

maxCount = 5
startFFT = 1125  # Between 0 and 2265
threshold = 20 # Proportional to batch size



f = []
file = open('data/the-oao-data.txt', 'r')
count = 0
lines = file.readlines()

for d in lines[startFFT:]:
    if count == maxCount:
        break
    d = d[:-2]
    d = d.strip()
    res = list(map(float, d.split(',')))
    f.extend(res)
    count += 1  
    
print(len(f))
print(count)

# Create a simple signal with two frequencies
dt = count*4/len(f)
t = np.arange(0,count*4,dt)

# plt.plot(t,f,color='c',linewidth=1.5,label='Noisy')
# plt.plot(t,f_clean,color='k',linewidth=2,label='Clean') 
# plt.xlim(t[0],t[-1])
# plt.legend()

## Compute the Fast Fourier Transform (FFT)
n = len(t)
fhat = np.fft.fft(f,n)                      # Compute the FFFT
PSD = fhat * np.conj(fhat) / n              # Power spectrum (power per freqeuncy)
freq = (1/(dt)) * np.arange(n)            # Create x-axis of frequencies
L = np.arange(1,n, dtype='int') # Only plot the first half of

## Filter out noise
print(len(PSD))
indices = [0] * len(PSD)                 
for i in range(20):
    indices[i] = 1   
    indices[-i] = 1                 
PSDClean = PSD * indices                    # Zero out all others
fhat = fhat * indices                       # Zero out all small coefficients
ffilt = np.fft.ifft(fhat)                   # Inverse transform for filtered signal


# C
# index_1 = [0]*len(fhat)
# index_2 = [0]*len(fhat)
# HELPER = []
# count = 0
#for element in PSD:
#     if element > threshold:
#        HELPER.append(count)
#    count += 1



#index_1[HELPER[0]] = 1
#index_2[HELPER[1]] = 1

#ffilt_1 = np.fft.ifft(fhat * index_1)  
#ffilt_2 = np.fft.ifft(fhat * index_2)  


# Plot results
fig,axs = plt.subplots(4,1)
plt.sca(axs[0])
plt.plot(t,f,color='c',linewidth=1.5,label='Signal with ripple')
plt.plot(t,max(ffilt)*np.ones(len(t)),color='r',linewidth=1.5,label='Max Amplitude of Clean Signal')
#plt.plot (t,f_clean,color='k',linewidth=2,label='Clean')
plt.xlim(t[0],t[-1])
plt.ylim(min(f),max(f)) 

plt.legend()

#plt.sca(axs[1])
#plt.plot(t,ffilt_1,color='c',linewidth=1.5,label='First Sinusoidal Siganl')
#plt.xlim(t[0],t[-1]) 
#plt.ylim(min(f),max(f)) 
#axs[1].set(xlabel='time [s]', ylabel='amplitude')
#plt.legend()

plt.sca(axs[1])
plt.plot(t,ffilt,color='c',linewidth=1.5,label=max(ffilt))
plt.xlim(t[0],t[-1]) 
plt.ylim(min(f),max(f)) 
axs[1].set(xlabel='time [s]', ylabel='amplitude')
plt.legend()

plt.sca(axs[3])
#plt.plot(freq[L],PSD[L],color='c',linewidth=1.5,label='Noisy')
plt.plot(np.arange(len(PSD[L])),PSD[L],color='r',linewidth=1.5,label='Noisy')
axs[3].set(xlabel='frequency [Hz|', ylabel='power')

plt.show()
