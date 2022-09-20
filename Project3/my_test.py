import scipy.io.wavfile as wav
from python_speech_features import mfcc
import matplotlib.pyplot as plt
import numpy as np
import soundfile 
from scipy import signal

audio_clip = "./data/nlpnd_projects/LibriSpeech/dev-clean/2428/83705/2428-83705-0000.flac"
sig, rate = soundfile .read(audio_clip, dtype='float32')
print(f"data dimension {np.shape(sig)}")
print(f"rate {rate}")
print(f"number of channels = 1")
length = len(sig) / rate
print(f"length = {length}s")


time = np.linspace(0., length, sig.shape[0])
# plt.plot(time, sig, label="Left channel")
# # plt.plot(time, sig[:, 1], label="Right channel")
# plt.legend()
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.show()

mfcc_data = mfcc(sig, rate, numcep=13)

# plt.figure(figsize=(15,5))
# # plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
# plt.imshow(mfcc_data, aspect='auto', origin='lower');
# print(mfcc_data.shape)
# plt.show()
def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


frequencies, times, spectrogram = log_specgram(sig, rate, window_size=20,
                                        step_size=10, eps=1e-10)
# frequencies, times, spectrogram = signal.spectrogram(sig, rate, 
#                                             nperseg = 180,
#                                             # nfft = 256,
#                                             window='hann',
#                                             # scaling= "spectrum",
#                                             mode =  "magnitude")
print(np.shape(spectrogram))
plt.pcolormesh(times, frequencies, spectrogram.T)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')




plt.show()