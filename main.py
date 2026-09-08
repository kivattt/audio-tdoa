# 2.wav is always shorter than 1.wav
# smaller file within larger file correlated from different microphones
import os
import sys
import time
import warnings
import scipy.io.wavfile as wavfile
from scipy import signal
from scipy import fft
import numpy as np
from numba import njit

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

# Copilot
def find_offset_correlate_fft(audio_1, audio_2, samplerate):
    # Parameters for STFT
    nperseg = 1024
    noverlap = nperseg // 2
    
    # Compute STFT
    f1, t1, Sxx1 = signal.spectrogram(audio_1, samplerate, nperseg=nperseg, noverlap=noverlap)
    f2, t2, Sxx2 = signal.spectrogram(audio_2, samplerate, nperseg=nperseg, noverlap=noverlap)

    # Normalize each frame to unit norm (makes it amplitude-invariant)
    Sxx1 = Sxx1 / np.sqrt(np.sum(Sxx1**2, axis=0, keepdims=True) + 1e-10)
    Sxx2 = Sxx2 / np.sqrt(np.sum(Sxx2**2, axis=0, keepdims=True) + 1e-10)
    
    # Use log-magnitude for better perceptual matching (humans perceive log-amplitude)
    Sxx1 = np.log1p(Sxx1)
    Sxx2 = np.log1p(Sxx2)
    
    # Re-normalize after log
    Sxx1 = Sxx1 / np.sqrt(np.sum(Sxx1**2, axis=0, keepdims=True) + 1e-10)
    Sxx2 = Sxx2 / np.sqrt(np.sum(Sxx2**2, axis=0, keepdims=True) + 1e-10)
    
    # Slide Sxx2 across Sxx1 and compute correlation
    num_frames_1 = Sxx1.shape[1]
    num_frames_2 = Sxx2.shape[1]
    num_positions = num_frames_1 - num_frames_2 + 1
    
    spectral_correlation = np.zeros(num_positions)
    
    for i in range(num_positions):
        window_1 = Sxx1[:, i:i+num_frames_2]
        spectral_correlation[i] = np.sum(window_1 * Sxx2)
    
    # Find peak
    frame_offset = np.argmax(spectral_correlation)
    offset_seconds = t1[frame_offset]
    offset_samples = int(offset_seconds * samplerate)
    
    # Pad to expected length
    expected_len = len(audio_1) - len(audio_2) + 1
    repeat_factor = expected_len // len(spectral_correlation) + 1
    spectral_correlation_padded = np.repeat(spectral_correlation, repeat_factor)[:expected_len]

    assert(expected_len == len(spectral_correlation_padded))
    return offset_samples / samplerate, spectral_correlation_padded

# Copilot
def find_offset_correlate_fft2(audio_1, audio_2, samplerate):
    # This algorithm requires both samplerates to be identical.
    # If they aren't, one of the audio clips needs oversampling/downsampling to match
    audio_1 = np.asarray(audio_1, dtype=np.float32)
    audio_2 = np.asarray(audio_2, dtype=np.float32)

    # Parameters for STFT
    nperseg = 512
    noverlap = nperseg * 0.75

    # Compute spectrogram magnitudes (centers of windows -> t arrays are frame-center times)
    _, t1, S1 = signal.spectrogram(audio_1, samplerate, nperseg=nperseg, noverlap=noverlap, mode='magnitude')
    _, _, S2 = signal.spectrogram(audio_2, samplerate, nperseg=nperseg, noverlap=noverlap, mode='magnitude')

    # Preprocessing: frame-wise normalization, log-magnitude, re-normalize (as you had)
    def preprocess(S):
        S = S / np.sqrt(np.sum(S**2, axis=0, keepdims=True) + 1e-10)
        S = np.log1p(S)
        S = S / np.sqrt(np.sum(S**2, axis=0, keepdims=True) + 1e-10)
        return S

    S1 = preprocess(S1)
    S2 = preprocess(S2)

    F, N1 = S1.shape
    _, N2 = S2.shape
    num_positions = N1 - N2 + 1
    assert num_positions > 0

    # Prepare reversed template across frequency axis
    B_rev = S2[:, ::-1]   # shape (F, N2)

    # FFT-based convolution across frequency bins:
    conv_len = N1 + N2 - 1
    # Use next power of two for speed (optional but often faster)
    L = 1 << int(np.ceil(np.log2(conv_len)))

    # Zero-pad per-frequency time-series to length L
    A_pad = np.zeros((F, L), dtype=np.float32)
    B_pad = np.zeros((F, L), dtype=np.float32)
    A_pad[:, :N1] = S1
    B_pad[:, :N2] = B_rev  # note: B_rev already reversed along its time axis

    # Batch rFFT along time axis for each frequency bin
    FA = np.fft.rfft(A_pad, n=L, axis=1)
    FB = np.fft.rfft(B_pad, n=L, axis=1)

    # Multiply per-frequency, sum across frequencies in frequency-domain, inverse rfft once
    prod_sum = np.sum(FA * FB, axis=0)          # shape (L_rfft,)
    conv_all = np.fft.irfft(prod_sum, n=L)     # length L (>= conv_len)

    # valid positions for sliding-correlation correspond to indices [N2-1 ... N1-1]
    start_idx = N2 - 1
    end_idx = N1  # python slice exclusive
    spectral_correlation = conv_all[start_idx:end_idx]  # length num_positions

    i_peak = np.argmax(spectral_correlation)
    coarse_offset_samples = t1[i_peak] * samplerate

    # Normal time-domain correlation around the coarse estimate
    def correlation(audio_1, audio_2, center, samplerate):
        refine_window_sec = 0.1 # How much around the center we want to time-correlate on
        half_win = refine_window_sec * samplerate

        seg_start = int(max(0, center - half_win))
        seg_end = int(min(len(audio_1), center + half_win + len(audio_2))) # ensure seg can contain audio_2

        # use FFT-based correlate for speed on moderate sizes
        corr = signal.correlate(audio_1[seg_start:seg_end], audio_2, mode='valid', method='fft')
        samples_peak_index = seg_start + np.argmax(corr)

        return samples_peak_index, corr

    center = coarse_offset_samples
    final_offset_samples, correlation = correlation(audio_1, audio_2, center, samplerate)

    offset_samples_int = int(round(final_offset_samples))
    expected_len = max(1, len(audio_1) - len(audio_2) + 1)
    repeat_factor = max(1, expected_len // len(spectral_correlation) + 1)
    spectral_correlation_padded = np.repeat(spectral_correlation, repeat_factor)[:expected_len]
    spectral_correlation_padded = np.pad(spectral_correlation_padded, (0, len(audio_2)))

    do_plot = False
    if do_plot:
        #plt.plot(correlation) # Normal time-domain correlation (centered around `center` variable)
        self.plot_it(data1, spectral_correlation_padded, offset_samples_int) # Frequency domain correlation

    return final_offset_samples / samplerate, spectral_correlation_padded

# Best so far, passes all my tests
def find_offset_hybrid(audio_1, audio_2, samplerate):
    sos = signal.butter(N=2, Wn=10, btype="highpass", fs=samplerate, output="sos")
    audio_1 = signal.sosfiltfilt(sos, audio_1, padtype=None)
    audio_2 = signal.sosfiltfilt(sos, audio_2, padtype=None)

    ans1, cc1 = find_offset_correlate_fft(audio_1, audio_2, samplerate)
    ans2, cc2 = find_offset_correlate_mul_polarity(audio_1, audio_2, samplerate)

    # Normalize both to [0, 1] range
    cc1_norm = cc1 / (np.max(np.abs(cc1)) + 1e-10)
    cc2_norm = cc2 / (np.max(np.abs(cc2)) + 1e-10)
    
    bias = 0.01  # 0 = fft, 1 = corr * polarity
    cc3 = (1 - bias) * cc1_norm + bias * cc2_norm

    ans3 = np.argmax(cc3)
    return ans3 / samplerate, np.abs(cc3)

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

@njit
def compute_mae_numba(audio_1, audio_2):
    out_len = len(audio_1) - len(audio_2)
    correlation = np.zeros(out_len, dtype=np.float32)
    for i in range(out_len):
        correlation[i] = -np.sum(np.abs(audio_1[i:i+len(audio_2)] - audio_2))
    return correlation
def find_offset_mae(audio_1, audio_2, samplerate):
    correlation = compute_mae_numba(audio_1, audio_2)
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
    correlation = np.abs(correlation)

    correlation = np.pow(correlation, 3.0) # HACK: hack to improve test results...

    offset_seconds = np.argmax(correlation) / samplerate
    return offset_seconds, correlation

def find_offset_correlate_mul_polarity(audio_1, audio_2, samplerate):
    sos = signal.butter(N=2, Wn=10, btype="highpass", fs=samplerate, output="sos")
    audio_1 = signal.sosfiltfilt(sos, audio_1, padtype=None)
    audio_2 = signal.sosfiltfilt(sos, audio_2, padtype=None)

    ans1, cc1 = find_offset_correlate(audio_1, audio_2, samplerate)
    ans2, cc2 = find_offset_polarity(audio_1, audio_2, samplerate)

    #cc2 = np.sqrt(cc2)
    #cc2 = 1.0 + np.multiply(0.01, cc2)

    cc3 = cc1 * cc2
    ans3 = np.argmax(cc3)

    #C = 0.00593565133070766 # HACK: the average offset diff error in my tests at some point
    C = 0
    return ans3 / samplerate - C, np.abs(cc3)

def just_show_audio1(audio_1, audio_2, samplerate):
    return 0.0, audio_1

def just_show_audio2(audio_1, audio_2, samplerate):
    return 0.0, audio_2

def show_mul_at_calculated_correlate_offset(audio_1, audio_2, samplerate):
    #offset_sec, _ = find_offset_correlate_mul_polarity(audio_1, audio_2, samplerate)
    offset_sec, _ = find_offset_correlate(audio_1, audio_2, samplerate)
    offset = int(offset_sec * samplerate)
    clip_1 = audio_1[offset:offset+len(audio_2)]
    clip_2 = audio_2
    clip_1 = clip_1 / np.sqrt(np.sum(clip_1 **2))
    clip_2 = clip_2 / np.sqrt(np.sum(clip_2 **2))
    mul = clip_1 * clip_2
    mul = np.abs(mul)
    return offset_sec, mul

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

def benchmark_algorithm(algo, algo_name, samplerate, long_len, short_len):
    audio_1 = np.random.uniform(-1.0, 1.0, long_len).astype(np.float32)
    audio_2 = np.random.uniform(-1.0, 1.0, short_len).astype(np.float32)

    start = time.time()
    algo(audio_1, audio_2, samplerate)
    duration = time.time() - start
    algo_name = "\x1b[0;32m" + algo_name + " \x1b[0;37m"
    print(algo_name.ljust(50, '-') + "\x1b[0m " + str(duration) + "s")

#algos = [find_offset_correlate, find_offset_fft, find_offset_lrms, find_offset_gccphat_4]
#algos_names = ["Correlation", "FFT Correlation", "Least RMS", "GCC PHAT"]
algos = [find_offset_correlate, find_offset_correlate_fft, find_offset_correlate_fft2, find_offset_correlate_mul_polarity, find_offset_hybrid]
algos_names = ["Correlation", "FFT Correlation (copilot)", "fft2", "Correlation * Polarity", "Hybrid"]

do_profiling = True
if do_profiling:
    samplerate = 48000
    long_len = 200 * samplerate # 200 sec
    short_len = 2 * samplerate # 2 sec

    print("")
    print("Samplerate: " + str(samplerate))
    print("Long time: " + str(long_len / samplerate) + "s")
    print("Short time: " + str(short_len / samplerate) + "s")
    print("")

    i = 0
    for algo in algos:
        algo_name = algos_names[i]
        benchmark_algorithm(algo, algo_name, samplerate, long_len, short_len)

        i += 1
    sys.exit(0)

do_write = False
if os.path.exists(output_correlation_file):
    confirmation = input("\x1b[0;31mOverwrite " + output_correlation_file + " ? (y/N): \x1b[0m")
    if confirmation == "y" or confirmation == "Y":
        print("\x1b[0;33mOverwriting " + output_correlation_file + " ...\x1b[0m")
        # Delete the output correlation file
        os.remove(output_correlation_file)
        do_write = True
else:
    print("\x1b[0;33mWriting plot data to " + output_correlation_file + " ...\x1b[0m")
    do_write = True

def output_and_return(string):
    print(string, end="")
    return string

output_list = []

i = 0
for algo in algos:
    algoName = algos_names[i]
    i += 1

    string = output_and_return("\x1b[34m" + algoName + "\x1b[0m\n")
    output_list.append(string)

    for directory in sorted(os.listdir(tests_folder)):
        expected_file = os.path.join(tests_folder, directory, "expected_offset_seconds.txt")
        audio_file_1 = os.path.join(tests_folder, directory, "1.wav")
        audio_file_2 = os.path.join(tests_folder, directory, "2.wav")

        (calculated_offset, correlation), samplerate = find_offset_in_files(audio_file_1, audio_file_2, algo)
        with open(expected_file, 'r') as file:
            expected_offset = file.read().strip()

        diff = calculated_offset - float(expected_offset)
        diff_meters = diff * 343

        string = output_and_return("Test \x1b[0;32m" + directory.rjust(25, ' ') + "\x1b[0m expected: " + str(expected_offset).ljust(9, ' ') + ", got: " + str(calculated_offset).ljust(19,' ') + ", diff: " + diff_color_background(diff) + str(diff) +  f" ({diff_meters:.2f}m)" + "\x1b[0m\n")
        output_list.append(string)

        # Append data to the output correlation file
        if do_write:
            append_data(algoName, directory, samplerate, calculated_offset, correlation)

if do_write and len(output_list) > 0:
    with open(output_correlation_file, "rb") as f:
        existing_binary_data = f.read()

    # Prepend the output correlation file with the output list
    with open(output_correlation_file, "rb+") as f:
        f.seek(0, 0)
        f.write("OUTPUT_IS_HERE".encode("utf-8"))

        output_len = 0
        for line in output_list:
            output_len += len(line.encode("utf-8"))

        output_len_bytes = output_len.to_bytes(8, byteorder='little', signed=False)
        f.write(output_len_bytes)

        for line in output_list:
            f.write(line.encode("utf-8"))

        f.write(existing_binary_data)
