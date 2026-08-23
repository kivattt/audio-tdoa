# 2.wav is always shorter than 1.wav
# smaller file within larger file correlated from different microphones
import os
import warnings
import scipy.io.wavfile as wavfile
from scipy import signal
from scipy import fft
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import hilbert
from scipy.signal import fftconvolve

import matplotlib
matplotlib.use('TkAgg')  # Must be called BEFORE importing pyplot
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

tests_folder = "tests"
output_correlation_file = "correlation_log.data"

def find_offset_in_files(audio_file_1, audio_file_2, algo):
    try:
        fs1, audio_1 = wavfile.read(audio_file_1)
        fs2, audio_2 = wavfile.read(audio_file_2)
    except WavFileWarning:
        pass

    assert fs1 == fs2
    return algo(audio_1, audio_2, fs1), fs1

def find_offset_correlate(audio_1, audio_2, samplerate):
    audio_2 = audio_2 / np.sqrt(np.sum(audio_2**2))
    audio_1 = audio_1 / np.sqrt(np.sum(audio_1**2))
    cross_correlation = signal.correlate(audio_1, audio_2, mode="valid")
    offset_samples = cross_correlation.argmax()
    #offset_samples = np.argmax(cross_correlation)
    return offset_samples / samplerate, np.abs(cross_correlation)

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

    correlation = np.array([])
    for i in range(len(audio_1_fft_chunks) - len(audio_2_fft_chunks)):
        my_sum = 0
        for j in range(len(audio_2_fft_chunks)):
            fft1 = audio_1_fft_chunks[i+j]
            fft2 = audio_2_fft_chunks[j]

            fft1_amplitudes = (2.0 / len(fft1)) * np.abs(fft1[:len(fft1)//2])
            fft2_amplitudes = (2.0 / len(fft2)) * np.abs(fft2[:len(fft2)//2])
            my_sum += np.correlate(fft1_amplitudes, fft2_amplitudes)

        correlation = np.append(correlation, my_sum)

    #plt.plot(correlation)
    #plt.show()

    peak_index = np.argmax(np.abs(correlation))

    offset_samples = peak_index * chunk_size
    return offset_samples / samplerate, np.repeat(correlation, chunk_size)

def find_offset_lrms(audio_1, audio_2, samplerate):
    decimate_scale = 8
    n_times = 2
    for i in range(n_times):
        audio_1 = signal.decimate(audio_1, q=decimate_scale)
        audio_2 = signal.decimate(audio_2, q=decimate_scale)

    decimate_scale = 8**n_times

    correlation = np.zeros(len(audio_1) - len(audio_2))

    audio_2_rms = np.sqrt(np.mean(audio_2 ** 2))

    for i in range(len(audio_1) - len(audio_2)):
        #section = audio_1[i:i+len(audio_2)].copy()
        section = audio_1[i:i+len(audio_2)].copy()
        section_rms = np.sqrt(np.mean(section ** 2))

        section *= audio_2_rms / section_rms # normalize the volume of the section to match section_rms

        rms_cancellation = np.sqrt(np.mean((section - audio_2_rms) ** 2))
        #correlation[i] = 1.0 / (1.0 + rms_cancellation)
        correlation[i] = rms_cancellation

    # HPF
    window = signal.tukey(len(correlation), alpha=0.1)

    sos = signal.butter(N=2, Wn=10, btype="highpass", fs=samplerate / decimate_scale, output="sos")
    correlation = signal.sosfiltfilt(sos, window * correlation, padtype=None)
    # LPF
    sos = signal.butter(N=2, Wn=4, btype="lowpass", fs=samplerate / decimate_scale, output="sos")
    correlation = signal.sosfiltfilt(sos, window * correlation, padtype=None)
    # ABS
    correlation = np.abs(correlation)

    offset_samples = np.argmin(correlation)
    correlation = np.repeat(correlation, decimate_scale)
    return offset_samples * decimate_scale / samplerate, correlation

def find_offset_gccphat_4(audio_1, audio_2, samplerate):
    # 1. Normalize the input signals exactly like your correlate function
    audio_1 = audio_1 / np.sqrt(np.sum(audio_1**2))
    audio_2 = audio_2 / np.sqrt(np.sum(audio_2**2))
    
    # 2. Total linear convolution length
    n = len(audio_1) + len(audio_2) - 1
    
    # 3. Compute FFTs
    X1 = np.fft.rfft(audio_1, n=n)
    X2 = np.fft.rfft(audio_2, n=n)
    
    # 4. Cross-power spectrum
    R = X1 * np.conjugate(X2)
    
    # 5. Apply PHAT whitening filter
    R_phat = R / (np.abs(R) + 1e-9)
    
    # 6. Return back to time domain (unshifted)
    # Here, index 0 is 0 delay, index 1 is +1 sample delay, etc.
    # The negative delays wrap around to the very end of the array.
    cc_full = np.fft.irfft(R_phat, n=n)
    
    # 7. Slice array to match mode="valid" length perfectly
    # This keeps only positive delays from 0 up to (len(audio_1) - len(audio_2))
    valid_length = len(audio_1) - len(audio_2) + 1
    cross_correlation = cc_full[0:valid_length]
    
    # 8. Find the peak position among positive delays
    offset_samples = cross_correlation.argmax()

    #return offset_samples / samplerate, np.abs(cross_correlation)
    return offset_samples / samplerate, cross_correlation

def find_offset_ssd(audio_1, audio_2, samplerate):
    N1 = len(audio_1)
    N2 = len(audio_2)
    out_len = N1 - N2
    sum_b2 = np.sum(audio_2 ** 2)
    cross_term = fftconvolve(audio_1, audio_2[::-1], mode='valid')[:out_len]
    quad_1 = np.insert(np.cumsum(audio_1 ** 2), 0, 0)
    sum_a2 = quad_1[N2:N2 + out_len] - quad_1[:out_len]
    correlation = 2 * cross_term - sum_a2 - sum_b2
    offset_seconds = np.argmax(correlation) / samplerate
    return offset_seconds, correlation

def find_offset_mae(audio_1, audio_2, samplerate):
    out_len = len(audio_1) - len(audio_2)
    windows = sliding_window_view(audio_1, window_shape=len(audio_2))[:out_len]
    correlation = np.zeros(out_len, dtype=np.float32)
    batch_size=5000
    for i in range(0, out_len, batch_size):
        end = min(i + batch_size, out_len)
        correlation[i:end] = -np.sum(np.abs(windows[i:end] - audio_2), axis=1)
    offset_seconds = np.argmax(correlation) / samplerate
    return offset_seconds, correlation

def find_offset_envelope(audio_1, audio_2, samplerate):
    N1 = len(audio_1)
    N2 = len(audio_2)
    out_len = N1 - N2
    env_1 = np.abs(hilbert(audio_1))
    env_2 = np.abs(hilbert(audio_2))
    sum_b2 = np.sum(env_2 ** 2)
    cross_term = fftconvolve(env_1, env_2[::-1], mode='valid')[:out_len]
    quad_1 = np.insert(np.cumsum(env_1 ** 2), 0, 0)
    sum_a2 = quad_1[N2:N2 + out_len] - quad_1[:out_len]
    correlation = 2 * cross_term - sum_a2 - sum_b2
    offset_seconds = np.argmax(correlation) / samplerate
    return offset_seconds, correlation

def find_offset_polarity(audio_1, audio_2, samplerate):
    out_len = len(audio_1) - len(audio_2)
    sign_1 = np.sign(audio_1)
    sign_2 = np.sign(audio_2)
    #correlation = fftconvolve(sign_1, sign_2[::-1], mode='valid')[:out_len]
    correlation = fftconvolve(sign_1, sign_2[::-1], mode='valid')[:out_len+1]
    offset_seconds = np.argmax(correlation) / samplerate
    return offset_seconds, correlation

def find_offset_correlate_mul_polarity(audio_1, audio_2, samplerate):
    ans1, cc1 = find_offset_correlate(audio_1, audio_2, samplerate)
    ans2, cc2 = find_offset_polarity(audio_1, audio_2, samplerate)

    cc3 = cc1 * cc2
    ans3 = np.argmax(cc3)

    return ans3 / samplerate, cc3

def diff_color_background(diff_value):
    redness = abs(diff_value)
    red = int(min(255, redness * 800))
    green = 0
    blue = 0
    return "\x1b[48;2;" + str(red) + ";" + str(green) + ";" + str(blue) + "m"

def append_data(algorithm_str, test_name_str, samplerate, calculated_offset, correlation):
    algorithm_str_len_bytes = len(algorithm_str).to_bytes(8, byteorder='little', signed=False)
    test_name_str_len_bytes = len(test_name_str).to_bytes(8, byteorder='little', signed=False)

    # Hardcoded float32 length (32 / 8 = 4 bytes)
    correlation_len_bytes = (4 * len(correlation)).to_bytes(8, byteorder='little', signed=False)

    with open(output_correlation_file, "ab") as f:
        f.write(samplerate.to_bytes(8, byteorder='little', signed=False))
        #f.write(float(calculated_offset).to_bytes(8, byteorder='little', signed=False))
        f.write(np.array([calculated_offset]).astype(np.float64))

        f.write(algorithm_str_len_bytes)
        f.write(algorithm_str.encode("utf-8"))

        f.write(test_name_str_len_bytes)
        f.write(test_name_str.encode("utf-8"))

        f.write(correlation_len_bytes)
        f.write(correlation.astype(np.float32))

# Delete the output correlation file
if os.path.exists(output_correlation_file):
    os.remove(output_correlation_file)

#algos = [find_offset_correlate, find_offset_fft, find_offset_lrms, find_offset_gccphat_4]
#algos_names = ["Correlation", "FFT Correlation", "Least RMS", "GCC PHAT"]
#algos = [find_offset_correlate, find_offset_ssd, find_offset_envelope, find_offset_polarity]
#algos_names = ["Correlation", "SSD", "Envelope", "Polarity"]
algos = [find_offset_correlate, find_offset_polarity, find_offset_correlate_mul_polarity]
algos_names = ["Correlation", "Polarity", "Correlation * Polarity"]

i = 0
for algo in algos:
    algoName = algos_names[i]
    i += 1

    print("\x1b[34m" + algoName + "\x1b[0m")

    for directory in os.listdir(tests_folder):
        expected_file = os.path.join(tests_folder, directory, "expected_offset_seconds.txt")
        audio_file_1 = os.path.join(tests_folder, directory, "1.wav")
        audio_file_2 = os.path.join(tests_folder, directory, "2.wav")

        (calculated_offset, correlation), samplerate = find_offset_in_files(audio_file_1, audio_file_2, algo)
        with open(expected_file, 'r') as file:
            expected_offset = file.read().strip()

        diff = calculated_offset - float(expected_offset)

        print("Test \x1b[0;32m" + directory.rjust(20, ' ') + "\x1b[0m expected: " + str(expected_offset).ljust(8, ' ') + ", got: " + str(calculated_offset).ljust(17,' '), ", diff: " + diff_color_background(diff) + str(diff) + "\x1b[0m")

        # Append data to the output correlation file
        append_data(algoName, directory, samplerate, calculated_offset, correlation)
