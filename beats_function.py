import numpy as np
from scipy.io import wavfile
import utilFunctions as UF
from math import *
from audio_to_STFT import audiotoSTFT
import matplotlib.pyplot as plt
import numpy.matlib
from scipy import signal
from audio_to_noveltyCurve import noveltyCurve
from noveltytoTempogram import NoveltyCurve_to_Tempogram

def peaklist_novelty(noveltyCurve):
    size_novelty = len(noveltyCurve)
    peak_novelty = np.zeros(size_novelty)
    

    if( noveltyCurve[0] > noveltyCurve[1]):
        peak_novelty[0] = 3.75
        

    if( noveltyCurve[size_novelty-1] > noveltyCurve[size_novelty -2]):
        peak_novelty[size_novelty-1] = 3.75
        

    for l in np.arange(1,size_novelty-2):

        if(noveltyCurve[l] > noveltyCurve[l+1] and noveltyCurve[l] > noveltyCurve[l-1]):
            peak_novelty[l] = 3.75
        
    return peak_novelty



def beats_out(noveltypeaks,BPM,tempogram,featureRate):

    maxindex_tempogram = np.argmax(abs(tempogram),0)
    #print maxindex_tempogram
    
    temp_noveltypeaks = noveltypeaks

    beats_output = np.zeros(len(temp_noveltypeaks))

    time_0 =( 1.0/BPM[int(maxindex_tempogram[0])])* 60
    #print time_0
    index = int(time_0 * featureRate)
    #print index

    for n in np.arange(len(temp_noveltypeaks)):

        if(index == n):
            if(temp_noveltypeaks[n] == 3.75):
                beats_output[n] = 1;
                #print '1'
                t_peakunderconsideration = n/featureRate
                n_peaktempo = np.floor(t_peakunderconsideration * 5)

                if( n_peaktempo > np.shape(tempogram)[1] -1):
                    n_peaktempo = np.shape(tempogram)[1] -1

                beattime_puc = (1.0/BPM[int(maxindex_tempogram[int(n_peaktempo)])])* 60
                t_next = t_peakunderconsideration + beattime_puc

                if(t_next> 30):
                    t_next = 30
                index_t_next = np.floor(t_next * featureRate)
                index = index_t_next
            else:
                index = index +1


    return beats_output







"""   
(Fs,audio) = UF.wavread('./data_wav/open_004.wav');
parameter = {};
parameter['fs'] = Fs
parameter['win_len'] = 1024.0 * parameter['fs'] / 22050.0
#print parameter['win_len']
parameter['stepsize']= 512 * parameter['fs']/22050.0
parameter['compressionC'] = 1000
noveltyC,f_rate = noveltyCurve(audio,Fs,parameter)
#print len(noveltyC)
peaks = peaklist_novelty(noveltyC)
#print peaks
parameterTempogram = {}
parameterTempogram['featureRate'] = f_rate#featureRate
parameterTempogram['tempoWindow'] = 8
parameterTempogram['BPM'] = np.arange(100,300)
parameterTempogram['stepsize'] = np.ceil(parameterTempogram['featureRate']/5)
temp,freq = NoveltyCurve_to_Tempogram(noveltyC,parameterTempogram)
peakinfo = peaklist_novelty(noveltyC)
output = beats_out(peakinfo,freq,temp,f_rate)

time_beats = np.array([])
for i in np.arange(len(output)):
    if(output[i] == 1):
        time_beats = np.append(time_beats,[i/f_rate])
#print time_beats
out_file = open('beats.txt','w')
for i in np.arange(len(time_beats)):
    out_file.write("%f\n"%time_beats[i])
    
#out_file.write(output)
out_file.close()


#print output
plt.figure
#plt.plot(noveltyC)
#plt.figure(2)
plt.plot(peakinfo,marker = 'x',color = 'b')
plt.plot(output,marker = 'x',color = 'r')
plt.show()
"""
