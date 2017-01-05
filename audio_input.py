"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""
"""
import pyaudio
import wave
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output1.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []
frames1= []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    data1 = int(data)
    print data1
    frames.append(data)
    frames1.append(data1)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

#print frames

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

wf = wave.open('outputnew.wav', 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames1))
wf.close()
"""
INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

import pyaudio
import numpy as np
import wave
from scipy.io.wavfile import write

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

p=pyaudio.PyAudio() # start the PyAudio class
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #uses default input device
#frames = np.array([])
frames1 = np.array([])
#frames2 = np.array([])
# create a numpy array holding a single read of audio data
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)): #to it a few times just to see
    
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    x = np.float32(data)/norm_fact[data.dtype.name]
    print x
    print np.shape(x)
    
    frames1 =np.append(frames1,x)
    #x *= INT16_FAC
    #x = np.int16(x)
    #frames2 =np.append(frames2,x)
    #print(data)
    #print x
    
    #frames = np.append(frames,data)

# close the stream gracefully
stream.stop_stream()
stream.close()
p.terminate()

#write('output3.wav',RATE,frames)
write('outputnew.wav',RATE,frames1)
#write('output5.wav',RATE,frames2)
"""
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output1.wavwf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(frames)
wf.close()
"""
