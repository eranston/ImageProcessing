import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import spectrogram
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write




def q1(audio_path) -> np.array:
    """
    :param audio_path: path to q1 audio file
    :return: return q1 denoised version
    """
    WINDOW_SIZE = 800
    WINDOW_HOP = 320
    sample_rate, samples = wavfile.read(audio_path)
    stft = stft_values(samples , WINDOW_SIZE , WINDOW_HOP)
    # draw spectogram before removal of noise freq
    temp = np.log(1+np.abs((stft)))
    plot_spectrogram(temp ,WINDOW_SIZE, WINDOW_HOP , 1)

    stft[226] = stft[226] -np.mean(stft[226][1:-10])
    stft[225] = stft[225] -np.mean(stft[225][1:-10])
    stft[224] = stft[224] -np.mean(stft[224][1:-10])

    # draw spectogram after removal of noise freq
    temp = np.log(1+np.abs((stft)))
    plot_spectrogram(temp ,WINDOW_SIZE, WINDOW_HOP , 2)

    # samples = istft_values(stft , WINDOW_SIZE , WINDOW_HOP)
    # WINDOW_SIZE = 200
    # WINDOW_HOP = 80
    # stft = stft_values(samples , WINDOW_SIZE , WINDOW_HOP)
    # stft_sepct_log_manipulated = np.log(1+np.abs((stft)))
    
    # stft_sepct_log_manipulated = np.log(1+np.abs((stft)))
    # plot_spectrogram(stft_sepct_log_manipulated[:WINDOW_SIZE//2,:] ,WINDOW_SIZE, WINDOW_HOP)
    # stft_sepct_log_manipulated[stft_sepct_log_manipulated < THRESHOLD] = 0
    # stft[stft_sepct_log_manipulated < THRESHOLD] = 0

    # plot_spectrogram(stft_sepct_log_manipulated[:WINDOW_SIZE//2,:] ,WINDOW_SIZE, WINDOW_HOP)
    wav_values = istft_values(stft , WINDOW_SIZE , WINDOW_HOP)
    return wav_values


def q2(audio_path) -> np.array:
    """
    :param audio_path: path to q2 audio file
    :return: return q2 denoised version
    """
    # read the file and parameters decleration
    WINDOW_SIZE = 400
    WINDOW_HOP = 100

    sample_rate, samples = wavfile.read(audio_path)
    # stft 
    stft = stft_values(samples , WINDOW_SIZE , WINDOW_HOP)

    # draw spectogram brefore removel
    stft_spec_log = np.log(1+np.abs((stft)))
    plot_spectrogram(stft_spec_log[:WINDOW_SIZE//2,:] ,WINDOW_SIZE, WINDOW_HOP,3)

    # remove all freq in array from starting index
    starting_index = 64
    last_index = 167
    freq_to_delete = range(57,63)
    stft = remove_freq(stft , starting_index ,last_index, freq_to_delete)


    # draw spectogram after removel
    stft_spec_log = np.log(1+np.abs((stft)))
    plot_spectrogram(stft_spec_log[:WINDOW_SIZE//2,:] ,WINDOW_SIZE, WINDOW_HOP,4)


    #transform back
    wav_values = istft_values(stft , WINDOW_SIZE , WINDOW_HOP)
    return wav_values

    
# function that get stft , array of freqencies to remove from the stft , and a starting index from 
# where to start to remove
def remove_freq(stft ,first_index,last_index , freq):
    for f in freq:
        stft[f] = np.concatenate([stft[f][:first_index] , [0 for i in range(first_index ,last_index)] , stft[f][last_index:]])
    return stft


# function that get the data of the audio and the sample rate and plot a graph of the freqencies
# of the audio
def plot_freq_audio(audio_data,sample_rate):
    # Perform FFT on the audio signal
    fft_result = np.fft.fft(audio_data)
    # Calculate the corresponding frequencies
    frequencies = np.fft.fftfreq(len(fft_result), d=1/sample_rate)
    # Plot the magnitude spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0,sample_rate , len(audio_data)), np.abs(fft_result))
    plt.title('FFT of Audio Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()




# function that create from stft calculate log(1+|F(u)|)(the log + 1 of the abs of all furia transform for
# each window)


# istft , back to audio data
def istft_values(stft_array , window_length , window_jump):
    return librosa.istft(stft_array , hop_length=window_jump , n_fft=window_length,win_length=window_length)



# calculate the stft of the audio
def stft_values(signal , window_length , window_jump):
    # Set parameters for the spectrogram
    S_scale = librosa.stft(signal, n_fft=window_length, hop_length=window_jump,win_length=window_length)
    return S_scale

# graph the audio signal
def graph_audio(signal , framerate):
    duration = len(signal) / framerate
    time = np.linspace(0., duration, len(signal))
    # Plot the audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal, color='blue')
    plt.title('Audio Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# function that get values of Y , samplerate , window hop length and create a spectogram
def plot_spectrogram(Y, sr, hop_length, n,y_axis="linear"  ):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.savefig(f"spectogram{n}.jpg")
    plt.show()



wav1 = q1("C:/Users/1eran/OneDrive/Desktop/ImageProcessing/ex2/q1.wav")
# wav2 = q2("C:/Users/1eran/OneDrive/Desktop/ImageProcessing/ex2/q2.wav")
# write("output1.wav", 4000 ,wav1)
# write("output2.wav", 4000 ,wav2)