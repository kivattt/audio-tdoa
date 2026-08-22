# 2.wav is always shorter than 1.wav
# smaller file within larger file correlated from different microphones
import os
import warnings
import scipy.io.wavfile as wavfile
from scipy import signal
from scipy import fft
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # Must be called BEFORE importing pyplot
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

tests_folder = "tests"

def find_offset_in_files(audio_file_1, audio_file_2, algo):
    try:
        fs1, audio_1 = wavfile.read(audio_file_1)
        fs2, audio_2 = wavfile.read(audio_file_2)
    except WavFileWarning:
        pass

    assert fs1 == fs2
    return algo(audio_1, audio_2, fs1)

def find_offset_correlate(audio_1, audio_2, samplerate):
    audio_2 = audio_2 / np.sqrt(np.sum(audio_2**2))
    audio_1 = audio_1 / np.sqrt(np.sum(audio_1**2))
    cross_correlation = signal.correlate(audio_1, audio_2, mode="valid")
    offset_samples = cross_correlation.argmax()
    #offset_samples = np.argmax(cross_correlation)
    return offset_samples / samplerate

# Maybe we should bias by loud amplitudes?
# So frequencies with loud amplitudes make up a bigger portion of the result
def correlate_two_fft_chunks(fft1, fft2):
    # Best
    fft1_amplitudes = (2.0 / len(fft1)) * np.abs(fft1[:len(fft1)//2])
    fft2_amplitudes = (2.0 / len(fft2)) * np.abs(fft2[:len(fft2)//2])
    return np.correlate(fft1_amplitudes, fft2_amplitudes)

    # Worse
    #return np.sum(fft1.real * fft2.real)

    # Phase information seems useless, just adds noise...
    #fft1_phase_list = np.angle(fft1[:len(fft1)//2])
    #fft2_phase_list = np.angle(fft2[:len(fft2)//2])
    #return np.correlate(fft1_phase_list, fft2_phase_list)

    # Even with amplitude, theres no reason to use phase since it just adds noise...
    #return np.correlate(fft1_amplitudes, fft2_amplitudes) * np.correlate(fft1_phase_list, fft2_phase_list)

# We could optimize for most "peakiness" in the correlation result by bruteforcing different chunk sizes.
#
# Or, maybe we could start high at like 100ms chunk size, trusting that to be an accurate but rough result,
# then lowering the chunk size to get better time resolution (only looking within a certain area of the certain 100ms result)
def find_offset_fft(audio_1, audio_2, samplerate):
    audio_1_fft_chunks = []
    audio_2_fft_chunks = []

    chunk_size_milliseconds = 100
    chunk_size = int(samplerate / (1000 / chunk_size_milliseconds))

    for i in range(int(len(audio_1) / chunk_size)):
        chunk = fft.fft(audio_1[i*chunk_size:i*chunk_size + chunk_size])
        audio_1_fft_chunks.append(chunk)

    for i in range(int(len(audio_2) / chunk_size)):
        chunk = fft.fft(audio_2[i*chunk_size:i*chunk_size + chunk_size])
        audio_2_fft_chunks.append(chunk)

    correlation = []
    for i in range(len(audio_1_fft_chunks) - len(audio_2_fft_chunks)):
        my_sum = 0
        for j in range(len(audio_2_fft_chunks)):
            fft1 = audio_1_fft_chunks[i+j]
            fft2 = audio_2_fft_chunks[j]

            fft1_amplitudes = (2.0 / len(fft1)) * np.abs(fft1[:len(fft1)//2])
            fft2_amplitudes = (2.0 / len(fft2)) * np.abs(fft2[:len(fft2)//2])
            my_sum += np.correlate(fft1_amplitudes, fft2_amplitudes)

        correlation.append(my_sum)

    peak = 0
    peak_index = 0
    for i in range(len(correlation)):
        if abs(correlation[i]) > peak:
            peak = abs(correlation[i])
            peak_index = i

    #plt.plot(correlation)
    #plt.show()

    offset_samples = peak_index * chunk_size
    return offset_samples / samplerate

# Kind of works, has an alright result for "tests/foreloopig"
def find_offset_gccphat_1(refsig, sig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT) method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # Zero out negative time delays (KIVA)
    lags = (np.arange(len(cc)) - max_shift) / (fs * interp)
    cc[lags < 0] = 0

    raw_peak_index = np.argmax(cc)
    shifted_peak_index = raw_peak_index - max_shift
    tau = shifted_peak_index / float(interp)

    # Plotting
    '''
    time_axis = (np.arange(len(cc)) - max_shift) / (fs * interp)
    time_offset = shifted_peak_index / (fs * interp)
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, cc, label='GCC-PHAT Coefficient', color='b')
    plt.axvline(time_offset, color='r', linestyle='--', label=f'Detected Delay: {time_offset:.4f}s')
    plt.title('GCC-PHAT Cross-Correlation Plot')
    plt.xlabel('Time Delay (seconds)')
    plt.ylabel('Correlation Amplitude')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()
    '''
    
    return tau / fs

# Doesn't work
def find_offset_gccphat_2(data1, data2, samplerate):
    if data1.size == 0 or data2.size == 0:
        return 0

    # Convert potential multi-channel input to mono and normalize dtype.
    ref = np.asarray(data1, dtype=np.float64)
    sig = np.asarray(data2, dtype=np.float64)

    ref_len = ref.shape[0]
    sig_len = sig.shape[0]
    fft_len = ref_len + sig_len - 1

    ref_fft = np.fft.rfft(ref, n=fft_len)
    sig_fft = np.fft.rfft(sig, n=fft_len)
    cross_spectrum = ref_fft * np.conj(sig_fft)

    magnitude = np.abs(cross_spectrum)
    magnitude[magnitude < 1e-15] = 1e-15
    correlation = np.fft.irfft(cross_spectrum / magnitude, n=fft_len)

    # Reorder to match integer lag range [-(len(sig)-1), len(ref)-1].
    correlation = np.concatenate((correlation[-(sig_len - 1):], correlation[:ref_len]))
    lags = np.arange(-(sig_len - 1), ref_len)
    lag = lags[np.argmax(np.abs(correlation))]
    return lag / samplerate

algos = [find_offset_correlate, find_offset_fft, find_offset_gccphat_1]

for algo in algos:
    print("\x1b[34m" + algo.__name__ + "\x1b[0m")

    for directory in os.listdir(tests_folder):
        expected_file = os.path.join(tests_folder, directory, "expected_offset_seconds.txt")
        audio_file_1 = os.path.join(tests_folder, directory, "1.wav")
        audio_file_2 = os.path.join(tests_folder, directory, "2.wav")

        calculated_offset = find_offset_in_files(audio_file_1, audio_file_2, algo)
        with open(expected_file, 'r') as file:
            expected_offset = file.read().strip()

        print("Test \x1b[0;32m" + directory.rjust(20, ' ') + "\x1b[0m expected: " + str(expected_offset) + ", got: " + str(calculated_offset))
