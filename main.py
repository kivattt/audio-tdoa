# smaller file within larger file correlated from different microphones
import os
import scipy.io.wavfile as wavfile
from scipy import signal
import numpy as np

tests_folder = "tests"

def find_offset_in_files(audio_file_1, audio_file_2):
    fs1, audio_1 = wavfile.read(audio_file_1)
    fs2, audio_2 = wavfile.read(audio_file_2)

    #assert fs1 == 44100 # hardcoded samplerate
    assert fs1 == fs2

    return find_offset(audio_1, audio_2, fs1)

def find_offset(audio_1, audio_2, samplerate):
    audio_2 = audio_2 / np.sqrt(np.sum(audio_2**2))
    audio_1 = audio_1 / np.sqrt(np.sum(audio_1**2))
    cross_correlation = signal.correlate(audio_1, audio_2, mode="valid")
    offset_samples = cross_correlation.argmax()
    #offset_samples = np.argmax(cross_correlation)
    return offset_samples / samplerate

def find_offset_gccphat(data1, data2):
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
    return lag / 44100 # hardcoded samplerate

for directory in os.listdir(tests_folder):
    expected_file = os.path.join(tests_folder, directory, "expected_offset_seconds.txt")
    audio_file_1 = os.path.join(tests_folder, directory, "1.wav")
    audio_file_2 = os.path.join(tests_folder, directory, "2.wav")

    calculated_offset = find_offset_in_files(audio_file_1, audio_file_2)
    with open(expected_file, 'r') as file:
        expected_offset = file.read().strip()

    print("Test \x1b[0;32m" + directory + "\x1b[0m expected: " + str(expected_offset) + ", got: " + str(calculated_offset))
