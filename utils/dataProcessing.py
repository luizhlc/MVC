import numpy as np
from scipy.fftpack import fft

def fill_the_gaps(data):
    ok = np.invert(np.isnan(data))
    xp = ok.ravel().nonzero()[0]
    fp = data[np.invert(np.isnan(data))]
    x  = np.isnan(data).ravel().nonzero()[0]
    data[np.isnan(data)] = np.interp(x, xp, fp)
    return data

def get_RMS(sig, f, n, w):
    steps = np.int_(np.floor(n/w))
    x_RMS = np.zeros((len(sig),steps))
    for i in range (0, steps):
        x_RMS[:,i] = np.sqrt(np.mean((sig[:, (i*w):((i+1)*w)]**2), axis=1))
    return x_RMS

def get_peak2peak(s_data):
    return np.max(s_data, axis=1)-np.min(s_data, axis=1)

def get_peak(s_data):
    return np.max(s_data, axis=1)

def get_crista(peak, rms_global):
    return peak/rms_global[:,0]

def apply_fft(sig, f, n):
    xf = np.linspace(0.0, (f/2.0), n//2)
    yf = fft(sig[:])
    yf = 2.0/n * np.abs(yf[:, :n//2])
    return xf, yf
