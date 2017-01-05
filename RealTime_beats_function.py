import pyaudio
import wave
import numpy as np
from scipy.io import wavfile
from math import *
#from audio_to_STFT import audiotoSTFT
import matplotlib.pyplot as plt
import numpy.matlib
from scipy import signal
from audio_to_noveltyCurveRT import noveltyCurveRT
from noveltytoTempogram import NoveltyCurve_to_Tempogram
import time


#norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}
start_time = time.time()
INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 45
#WAVE_OUTPUT_FILENAME = "output1.wav"

Fs =44100
parameter = {};
parameter['fs'] = Fs
parameter['win_len'] = 1024.0 * parameter['fs'] / 22050.0
parameter['stepsize']= 512 * parameter['fs']/22050.0
parameter['compressionC'] = 1000

parameterTempogram = {}
#parameterTempogram['featureRate'] = f_rate#featureRate
parameterTempogram['tempoWindow'] = 0.7
parameterTempogram['BPM'] = np.arange(100,300)
#parameterTempogram['stepsize'] = np.ceil(parameterTempogram['featureRate']/5)

noveltyCurve_RT = np.zeros([1,1024]);


p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames1 = np.array([])
frames = []
yout = np.array([])
NoveltyCurve_accum = np.array([])
T_accum = np.array([])
prev_bandData = {}
#NoveltyCurve_accum_peaks = np.array([])
tempo2_column =0
frame_num =0
t_elapsed =0
t_peak =np.array([])
peak_index = 0
tempo_index = 0
cnt = 0
tempogram =np.zeros([200,1])

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    x = np.float32(data)/norm_fact[data.dtype.name]
    

    yout = np.append(yout,x)
    frame_num= frame_num +1
    k = len(yout)
    t_elapsed = k/Fs

    (noveltyCurve_rt,featureRate,prev_bandData) = noveltyCurveRT(yout[k-2048:k],Fs,parameter,prev_bandData,frame_num)
    offset = len(noveltyCurve_rt)
    #print offset
    index = len(NoveltyCurve_accum)
    NoveltyCurve_accum = np.append(NoveltyCurve_accum,noveltyCurve_rt)
    if (frame_num ==2):
        NoveltyCurve_accum_peaks = np.zeros(int(RECORD_SECONDS*featureRate))
        if(NoveltyCurve_accum[peak_index]>NoveltyCurve_accum[peak_index+1]):
            NoveltyCurve_accum_peaks[0] = 3.75
        peak_index =1

        for n1 in np.arange(len(NoveltyCurve_accum)-1):
                            if(NoveltyCurve_accum[n1] > NoveltyCurve_accum[n1-1] and NoveltyCurve_accum[n1] >NoveltyCurve_accum[n1+1]):
                                NoveltyCurve_accum_peaks[n1] = 3
        if(NoveltyCurve_accum[len(NoveltyCurve_accum)-1]>NoveltyCurve_accum[len(NoveltyCurve_accum)-2]):
            NoveltyCurve_accum_peaks[0] = 3.75
            
    elif frame_num >2:

        for n1 in np.arange(index-1,len(NoveltyCurve_accum)-1):

            if(NoveltyCurve_accum[n1] > NoveltyCurve_accum[n1-1] and NoveltyCurve_accum[n1] >NoveltyCurve_accum[n1+1]):
                NoveltyCurve_accum_peaks =3
    parameterTempogram['featureRate'] = featureRate
    parameterTempogram['stepsize'] = np.ceil(parameterTempogram['featureRate']/10)
    index1 = len(NoveltyCurve_accum)-1

    if(frame_num%30 == 0):
        #print np.shape(NoveltyCurve_accum)

        (tempogram_fourier,T,BPM,t_rate) = NoveltyCurve_to_Tempogram(NoveltyCurve_accum[int(tempo_index):index],parameterTempogram)
        val = tempogram_fourier

        if t_elapsed>2:
            tempo_index = index1-offset- np.ceil(2*featureRate)
            new_tempoinfo = tempogram_fourier[:,tempo2_column:np.shape(tempogram_fourier)[1]]
            #print np.shape(new_tempoinfo)
            #print np.shape(tempogram)
            tempogram = np.hstack((tempogram,new_tempoinfo))
            #tempogram[:,l3:l3+np.shape(new_tempoinfo)[1]] = tempogram_fourier[:,tempo2_column:np.shape(tempogram_fourier)[1]]
            #l3 = l3 + np.shape(tempogram_fourier)[1];
            tempo2_column = np.shape(tempogram_fourier)[1]-5
            parameterTempogram['tempoWindow'] = 1;
            val = tempogram;
        T_accum = np.append(T_accum,T)
        cnt = cnt +1;
        maxindex_tempogram = np.argmax(abs(val),0)


    if frame_num ==2:
        beats_out = 1* (NoveltyCurve_accum_peaks>0)
        #print NoveltyCurve_accum
        #print beats_out
        #print sum(NoveltyCurve_accum_peaks)

        for n2 in np.arange(len(beats_out)):

            if beats_out[n2] == 1:
                print "beat detected"
                previous_beat_peak = n2;
                t_peak = np.append(t_peak,n2/featureRate)
                #ind_t_peak = ind_t_peak +1
    
                
    if cnt>1:
        peak_under_consideration = previous_beat_peak
        t_peak_under_consideration = (previous_beat_peak -1)/ featureRate

        npeaktempo = np.floor(t_peak_under_consideration*t_rate)+1
        if npeaktempo< np.shape(val)[1]:
            beattime_puc = (1/BPM[int(maxindex_tempogram[int(npeaktempo)])])*60
            t_next = t_peak_under_consideration+ beattime_puc
            t_peak = np.append(t_peak,t_next)
            index_t_next = np.floor(t_next * featureRate +1)
            beats_out[int(index_t_next)] = 1
            print "beat detected"
            previous_beat_peak = index_t_next;
    
    
    #print data1
    #print x
    #print np.shape(x)
    #print np.shape(frames1)
    #frames1 =np.append(frames1,yout)
    
    

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

stream.stop_stream()
stream.close()
p.terminate()

wavfile.write('output_final.wav',RATE,yout)
print("--- %s seconds ---" % (time.time() - start_time))
print "Time Stamps of Beats"
print t_peak
print "Total Number of Beats"
print np.shape(t_peak)
#print frames
#plt.figure
#plt.pcolormesh(np.arange(np.shape(tempogram)[1]),BPM,abs(tempogram))
#plt.show()



