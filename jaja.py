import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = [12, 10] 
plt.rcParams.update({'font.size': 18})

# Create a simple signal with two frequencies
dt = 0.001
t = np.arange(0,0.4,dt)
f = 3*np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*140*t) # Sum of 2 frequencies
f_clean = f
#f = f + 1.5*np.random.randn(len(t)) # Add some noise

# plt.plot(t,f,color='c',linewidth=1.5,label='Noisy')
# plt.plot(t,f_clean,color='k',linewidth=2,label='Clean') 
# plt.xlim(t[0],t[-1])
# plt.legend()

## Compute the Fast Fourier Transform (FFT)
n = len(t)
fhat = np.fft.fft(f,n)                      # Compute the FFFT
PSD = fhat * np.conj(fhat) / n              # Power spectrum (power per freqeuncy)
freq = (1/(dt*n)) * np.arange(n)            # Create x-axis of frequencies
L = np.arange(1,np.floor(n/2), dtype='int') # Only plot the first half of

## Filter out noise
threshold = 1
indices = PSD > threshold                   # Find frequencies with high power
PSDClean = PSD * indices                    # Zero out all others
fhat = fhat * indices                       # Zero out all small coefficients
ffilt = np.fft.ifft(fhat)                   # Inverse transform for filtered signal


# C
index_1 = [0]*len(fhat)
index_2 = [0]*len(fhat)
HELPER = []
count = 0
for element in PSD:
    if element > threshold:
        HELPER.append(count)
    count += 1

index_1[HELPER[0]] = 1
index_2[HELPER[1]] = 1

ffilt_1 = np.fft.ifft(fhat * index_1)  
ffilt_2 = np.fft.ifft(fhat * index_2)  

# Plot results
fig,axs = plt.subplots(4,1)
plt.sca(axs[0])
plt.plot(t,f,color='c',linewidth=1.5,label=max(f))
plt.plot (t,f_clean,color='k',linewidth=2,label='Clean')
plt.xlim(t[0],t[-1]) 
plt.ylim(min(f),max(f)) 

plt.legend()

plt.sca(axs[1])
plt.plot(t,ffilt_1+ffilt_2,color='c',linewidth=1.5,label=max(ffilt_1))
plt.xlim(t[0],t[-1]) 
plt.ylim(min(f),max(f)) 
axs[1].set(xlabel='time [s]', ylabel='amplitude')
plt.legend()

plt.sca(axs[2])
plt.plot(t,ffilt,color='c',linewidth=1.5,label=max(ffilt))
plt.xlim(t[0],t[-1]) 
plt.ylim(min(f),max(f)) 
axs[2].set(xlabel='time [s]', ylabel='amplitude')
plt.legend()

plt.sca(axs[3])
plt.plot(freq[L],PSD[L],color='c',linewidth=1.5,label='Noisy')
plt.plot(freq[L],PSDClean[L],color='r',linewidth=1.5,label='Noisy')
plt.xlim(freq[L[0]],freq[L[-1]]) 
axs[3].set(xlabel='frequency [Hz]', ylabel='power')

plt.show()






