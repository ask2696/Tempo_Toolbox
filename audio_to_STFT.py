import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def audiotoSTFT(f_audio,parameter):
    #use dictionary for parameter

    windowLength = len(parameter['StftWindow']);

    parameter['coefficientRange'] = np.array([1,np.floor(windowLength/2)+1])

    stepsize = parameter['stepsize'];
    featureRate = parameter['fs']/(stepsize);
    wav_size = len(f_audio);
    win = parameter['StftWindow'];
    first_win = int(np.floor(windowLength/2));
    num_frames = int(np.ceil(wav_size/stepsize));
    
    num_coeffs = int(parameter['coefficientRange'][1]-parameter['coefficientRange'][0]+1);
    
    #zerosToPad = max(0,parameter.nFFT - windowLength);
    #f_spec = zeros(num_coeffs,num_frames);
    f_spec = np.zeros([num_coeffs,num_frames])
    #print np.shape(f_spec)

    
    frame = np.arange(windowLength) - first_win 
    #print frame


    for n in np.arange(num_frames):
        
        numZeros = sum(frame<0)
        #print numZeros
        numVals = sum(frame>-1)
        #print numVals

        if numZeros > 0:
            x = np.append(np.zeros([numZeros,1]),f_audio[np.arange(numVals)])
            #print x
            x = np.multiply(x,win)
            #print x

        elif frame[len(frame)-1]> wav_size:
            x = np.append(f_audio[np.arange(frame[0],wav_size).astype(int)],np.zeros([int(windowLength -(wav_size -frame[0])),1]))
            x = np.multiply(x,win)
                          
        else:
            x = f_audio[frame.astype(int)] ####
            x = np.multiply(x,win)

        Xs = fft(x)
        index  = np.arange(parameter['coefficientRange'][0]-1,parameter['coefficientRange'][1])
        index = index.astype(int)
        f_spec[:,n] = abs(Xs[index])
        frame = frame+stepsize

    t = np.arange(np.shape(f_spec)[1]) * float(stepsize)/parameter['fs']
    freq = (np.arange(np.floor(windowLength/2)+1)/ (np.floor(windowLength/2)))*(parameter['fs']/2)
    freq = freq[np.arange(parameter['coefficientRange'][0]-1,parameter['coefficientRange'][1]).astype(int)]
    """
    plt.figure(1)
    #plt.subplot(211)
    plt.pcolormesh(t,freq,20*np.log10(f_spec))
    plt.show()
    """

    return (f_spec,featureRate,freq,t)  #[f_spec,featureRate,f,t]
            

        


