import numpy as np
from scipy.io import wavfile
import utilFunctions as UF
from math import *
from audio_to_STFT import audiotoSTFT
import matplotlib.pyplot as plt
import numpy.matlib
from scipy import signal



def myhann(n):
    res =  0.5 -0.5 * np.cos(2*pi*(np.arange(n)/(n-1)))                         
    return res



def noveltyCurve(f_audio,Fs):
    
    parameter = {};
    parameter['fs'] = Fs
    parameter['win_len'] = 1024.0 * parameter['fs'] / 22050.0
    #print parameter['win_len']
    parameter['stepsize']= 512 * parameter['fs']/22050.0
    parameter['compressionC'] = 1000

    parameter['StftWindow'] = myhann(parameter['win_len'])

    #print parameter['StftWindow']
    #[specData,featureRate,f,t] = audio_to_spectrogram_via_STFT(f_audio,parameter);

    (specData,featureRate,freq,t) = audiotoSTFT(f_audio,parameter)
    #print specData
    #print output

    #plt.figure(1)
    #plt.subplot(211)
    #plt.pcolormesh(t,freq,20*np.log10(specData))
    #plt.show()


    specData = specData/(specData.max());
    thresh = -74; # dB ???
    thresh = pow(10,(thresh/20));
    specData = thresh * (specData<= thresh) + specData *(specData >thresh);
    #print specData
    #plt.figure(2)
    #plt.subplot(212)
    #plt.pcolormesh(t,freq,20*np.log10(specData))
    #plt.show()

    bands = np.array([[0,500],[500,1250],[1250,3125],[3125,7812.5],[7182.5,np.floor(parameter['fs'])]])
    compression_C = parameter['compressionC']
    bandNoveltyCurves = np.zeros([np.shape(bands)[0],np.shape(specData)[1]])

    for band in np.arange(np.shape(bands)[0]):
                     bins = np.round(bands[band,:]/ (parameter['fs']/parameter['win_len']))
                     bins = np.maximum(1,bins)
                     bins = np.minimum(round(parameter['win_len']/2)+1,bins)


                     bandData = specData[int(bins[0]):int(bins[1]),:]
                     bandData_visual = specData[int(bins[0]):int(bins[1]),:]

                     bandData = np.log(1+ bandData * compression_C)/(np.log(1+ compression_C))

                     bandData_visual = np.log(1+ bandData * compression_C)/(np.log(1+ compression_C))
                     #print bandData
                     #print np.shape(bandData)

                     diff = np.diff(bandData)
                     #print diff
                     #print np.shape(diff)
                     diff = diff*(diff>0)

                     
                     add = np.array([bandData[:,0]]).T
                     

                     bandDiff = np.hstack((add,diff))
                     #print bandDiff
                     #print np.shape(bandDiff)

                     """
                     diff_len = 0.3
                     diff_len = max(np.ceil(diff_len* parameter['fs']/ parameter['stepsize']),5)
                     diff_len = 2 * round(diff_len/2)+1
                     mat = np.append(-1* np.ones([floor(diff_len/2),1]),np.append([0],np.ones([floor(diff_len/2),1])))
                     diff_filter = np.multiply(myhann(diff_len),mat)
                     diff_filter = diff_filter[:].transpose()
                     #[repmat(bandData(:,1),1,floor(diff_len/2)),bandData,repmat(bandData(:,end),1,floor(diff_len/2))]
                     filter2_2arg = np.append(np.matlib.repmat(bandData[:,1],1,int(floor(diff_len/2))),np.append(bandData,np.matlib.repmat(bandData[:,np.shape(bandData)[1]-1],1,int(floor(diff_len)/2))))
                     bandDiff = signal.convolve2d(diff_filter,filter2_2arg,mode ='same')
                     bandDiff = bandDiff * (bandDiff > 0)
                     bandDiff = bandDiff[:,int(np.floor(diff_len/2)):(np.shape(bandData)[1]-1-np.floor(diff_len/2)-1)]
                     bandDiff_visual[bins[0]:bins[2],:]= bandDiff[:,int(floor(diff_len/2)):np.shape(bandDiff)-1-int(floor(diff_len/2))-1]
                     """

                     
                     noveltyCurve = np.sum( bandDiff,0)
                     #print noveltyCurve
                     bandNoveltyCurves[band,:] = noveltyCurve
    
    #print np.shape(bandNoveltyCurves)
    NoveltyCurve = np.sum(bandNoveltyCurves,0)/(np.shape(bandNoveltyCurves)[0])
    #plt.figure(2)
    #plt.plot((np.arange(float(len(NoveltyCurve)))/featureRate),NoveltyCurve)
    #implement Local Avg
    local_avg = sum(NoveltyCurve)/ len(NoveltyCurve)
    NoveltyCurve = NoveltyCurve - local_avg
    NoveltyCurve = NoveltyCurve *(NoveltyCurve>0)
    #plt.figure(3)
    #plt.plot((np.arange(float(len(NoveltyCurve)))/featureRate),NoveltyCurve)
    #plt.show()
    return (NoveltyCurve,featureRate)
                                          
                     
#(Fs,audio) = UF.wavread('./data_wav/open_004.wav');
#noveltyCurve(audio,Fs)
                     
                                                     

                     
                     



    
