import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

output_correlation_file = "correlation_log.data"

@dataclass
class Correlation:
    samplerate: int
    algorithm: str
    #test_name: str
    calculated_offset: float
    correlation: np.ndarray

# Keys are test names
correlations = {}

with open(output_correlation_file, 'rb') as f:
    # Read loop
    while True:
        samplerate_bytes = f.read(8)
        if not samplerate_bytes:
            break

        samplerate = int.from_bytes(samplerate_bytes, byteorder='little', signed=False)
        calculated_offset = float(np.frombuffer(f.read(8), dtype=np.float64)[0])

        algorithm_str_len = int.from_bytes(f.read(8), byteorder='little', signed=False)
        algorithm_str = f.read(algorithm_str_len).decode("utf-8")

        test_name_str_len = int.from_bytes(f.read(8), byteorder='little', signed=False)
        test_name_str = f.read(test_name_str_len).decode("utf-8")

        correlation_len = int.from_bytes(f.read(8), byteorder='little', signed=False)
        correlation = np.frombuffer(f.read(correlation_len), dtype=np.float32) # float32 hardcoded

        correlations.setdefault(test_name_str, [])
        correlations[test_name_str].append(Correlation(
            samplerate = samplerate,
            algorithm = algorithm_str,
            #test_name = test_name_str,
            calculated_offset = calculated_offset,
            correlation = correlation,
        ))

for test_name, correlation_list in correlations.items():
    #fig, axes = plt.subplots(nrows=len(correlation_list), ncols=1, figsize=(8,4.5*len(correlation_list)), squeeze=False)
    fig, axes = plt.subplots(nrows=len(correlation_list), ncols=1, figsize=(8,3*len(correlation_list)), squeeze=False)
    axes = axes.flatten()

    tests_folder = "tests"
    expected_file = os.path.join(tests_folder, test_name, "expected_offset_seconds.txt")
    with open(expected_file, 'r') as file:
        expected_offset = float(file.read().strip())

    i = 0
    for c in correlation_list:
        time_axis = np.arange(len(c.correlation)) / c.samplerate
        axes[i].set_xlabel("Time (seconds)")
        axes[i].set_title("Test \"" + test_name + "\", algorithm: " + c.algorithm)

        axes[i].plot(time_axis, c.correlation)

        axes[i].axvline(x=expected_offset, color='green', linestyle='solid', linewidth=6, alpha=0.4)

        #computed_offset = np.argmax(np.abs(c.correlation)) / c.samplerate
        axes[i].axvline(x=c.calculated_offset, color='red', linestyle='dotted', linewidth=6, alpha=0.4)
        i += 1

    fig.tight_layout(pad=0.1)
    plt.show()
