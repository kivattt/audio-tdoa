// In the tests folder, 2.raw is always shorter than 1.raw
// We assume all audio files are 32 bit floating point raw data

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <dirent.h>

//#define DEBUGGING

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

typedef size_t (*Algorithm)(float*, size_t, float*, size_t);

void print_colored_symbol(float x) {
	float red = 1.0 - x * 6;

	red = pow(red, 1.0 / 2.2);
	red = MIN(1.0, MAX(0.0, red));

	uint8_t r = 255.0 * red;
	printf("\x1b[48;2;%i;%i;%imX\x1b[0m", r, 0, 0);
}

// Returns the first highest absolute peak
size_t find_abs_peak_index(float *audio, size_t sample_count) {
	float highest = 0;
	size_t out = 0;

	for (int i = 0; i < sample_count; i++) {
		float sample = fabs(audio[i]);
		if (sample > highest) {
			out = i;
			highest = sample;
		}
	}

	return out;
}

void compute_hpf(float *audio, size_t sample_count, float cutoff_hertz, int sample_rate) {
	float cutoff = cutoff_hertz * M_PI / (float)sample_rate;
	float thing = 0;

	for (int i = 0; i < sample_count; i++) {
		thing += (audio[i] - thing) * cutoff;
		audio[i] -= thing;
	}
}

void compute_lpf(float *audio, size_t sample_count, float cutoff_hertz, int sample_rate) {
	float cutoff = cutoff_hertz * M_PI / (float)sample_rate;
	float thing = 0;

	for (int i = 0; i < sample_count; i++) {
		thing += (audio[i] - thing) * cutoff;
		audio[i] = thing;
	}
}

// Remember to free() the returned pointer
float *compute_zero_crossings(float *audio, size_t sample_count) {
	float *out = malloc(sample_count * sizeof(float));
	out[0] = 0.0;

	float zero_crossing_value = 1.0;

	float last_sample = audio[0];
	for (int i = 1; i < sample_count; i++) {
		float sample = audio[i];

		float output = 0;
		if (last_sample > 0) {
			if (sample <= 0) {
				output = zero_crossing_value;
			}
		} else if (last_sample < 0) {
			if (sample >= 0) {
				output = zero_crossing_value;
			}
		}

		out[i] = output;
		last_sample = sample;
	}

	return out;
}

float compute_weird_sum(float *audio_1, size_t audio_1_sample_count, float *audio_2, size_t audio_2_sample_count) {
	assert(audio_1_sample_count == audio_2_sample_count);

	float sum = 0;
	for (int i = 0; i < audio_1_sample_count; i++) {
		//sum += audio_1[i] * audio_2[i];
		if ((audio_1[i] != 0) && audio_1[i] == audio_2[i]) {
			sum += 1.0;
		}
	}
	return fabs(sum);
}

void compute_auto_scale(float *audio, size_t sample_count) {
	float lowest = INFINITY;
	float highest = 0;
	for (int i = 0; i < sample_count; i++) {
		lowest = MIN(lowest, audio[i]);
		highest = MAX(highest, audio[i]);
	}

	for (int i = 0; i < sample_count; i++) {
		audio[i] = (audio[i] - lowest) / (highest - lowest);
	}
}

size_t find_offset_zero_crossings(float *audio_1, size_t audio_1_sample_count, float *audio_2, size_t audio_2_sample_count) {
	compute_hpf(audio_1, audio_1_sample_count, 20 /* cutoff in hertz */, 48000 /* sample rate */);
	compute_hpf(audio_2, audio_2_sample_count, 20 /* cutoff in hertz */, 48000 /* sample rate */);
	float *zero_crossings_1 = compute_zero_crossings(audio_1, audio_1_sample_count);
	float *zero_crossings_2 = compute_zero_crossings(audio_2, audio_2_sample_count);
	//compute_lpf(zero_crossings_1, audio_1_sample_count, 5, 48000);
	//compute_lpf(zero_crossings_2, audio_2_sample_count, 5, 48000);

	float *out = malloc(audio_1_sample_count * sizeof(float));
#pragma omp parallel for
	for (int i = 0; i < audio_1_sample_count - audio_2_sample_count; i++) {
		out[i] = compute_weird_sum(&zero_crossings_1[i], audio_2_sample_count, zero_crossings_2, audio_2_sample_count);
	}

#ifdef DEBUGGING
	compute_auto_scale(out, audio_1_sample_count - audio_2_sample_count);
	for (int i = 0; i < audio_1_sample_count; i++) {
		int _ = write(1, &out[i], sizeof(float));
	}
#endif

	free(zero_crossings_1);
	free(zero_crossings_2);
	size_t index = find_abs_peak_index(out, audio_1_sample_count);
	free(out);
	return index;
}

size_t temporary(float *audio_1, size_t audio_1_sample_count, float *audio_2, size_t audio_2_sample_count) {
	float *out = malloc(audio_1_sample_count * sizeof(float));
	compute_hpf(audio_1, audio_1_sample_count, 20, 44100);
	compute_lpf(audio_1, audio_1_sample_count, 40, 44100);
	compute_lpf(audio_1, audio_1_sample_count, 40, 44100);

#define N 8192
	for (int i = 0; i < audio_1_sample_count - audio_2_sample_count; i += N) {
		float zcr = 0;
		float last_sample = audio_1[i];
		for (int j = 1; j < N; j++) {
			float sample = audio_1[i+j];
			float score = fabs(sin(sample) - sin(last_sample)) / 2;
			last_sample = sample;
			zcr += score;
		}

		zcr /= N;
		printf("%f\n", zcr);

		for (int j = 0; j < N; j++) {
			out[i+j] = zcr;
		}
	}

	compute_auto_scale(out, audio_1_sample_count - audio_2_sample_count);
	for (int i = 0; i < audio_1_sample_count; i++) {
		int _ = write(1, &out[i], sizeof(float));
	}

	free(out);
	return 0;
}

size_t find_offset_correlation(float *audio_1, size_t audio_1_sample_count, float *audio_2, size_t audio_2_sample_count) {
	float *out = malloc(audio_1_sample_count * sizeof(float));

#pragma omp parallel for
	for (int i = 0; i < audio_1_sample_count - audio_2_sample_count; i++) {
		float sum = 0;
		for (int j = 0; j < audio_2_sample_count; j++) {
			sum += audio_1[i+j] * audio_2[j];
		}
		out[i] = sum;
	}

	size_t index = find_abs_peak_index(out, audio_1_sample_count - audio_2_sample_count);
	free(out);
	return index;
}

size_t find_offset_correlation_hpf(float *audio_1, size_t audio_1_sample_count, float *audio_2, size_t audio_2_sample_count) {
	float *out = malloc(audio_1_sample_count * sizeof(float));
	compute_hpf(audio_1, audio_1_sample_count, 20, 48000);
	compute_hpf(audio_2, audio_2_sample_count, 20, 48000);
	compute_lpf(audio_1, audio_1_sample_count, 5, 48000);
	compute_lpf(audio_2, audio_2_sample_count, 5, 48000);

#pragma omp parallel for
	for (int i = 0; i < audio_1_sample_count - audio_2_sample_count; i++) {
		float sum = 0;
		for (int j = 0; j < audio_2_sample_count; j++) {
			sum += audio_1[i+j] * audio_2[j];
		}
		out[i] = sum;
	}

	size_t index = find_abs_peak_index(out, audio_1_sample_count - audio_2_sample_count);
	free(out);
	return index;
}

float rms(float *audio, size_t sample_count) {
	float out = 0;
	for (int i = 0; i < sample_count; i++) {
		out += fabs(audio[i] * audio[i]);
	}
	return sqrt(out / (float)sample_count);
}

size_t find_offset_correlation_volume_normalized(float *audio_1, size_t audio_1_sample_count, float *audio_2, size_t audio_2_sample_count) {
	float *out = malloc(audio_1_sample_count * sizeof(float));

	float audio_2_rms = rms(audio_2, audio_2_sample_count);

#pragma omp parallel for
	for (int i = 0; i < audio_1_sample_count - audio_2_sample_count; i++) {
		float sum = 0;

		float audio_1_rms = rms(&audio_1[i], audio_2_sample_count);
		float gain = audio_2_rms / audio_1_rms;

		for (int j = 0; j < audio_2_sample_count; j++) {
			sum += (gain * audio_1[i+j]) * audio_2[j];
		}
		out[i] = sum;
	}

	size_t index = find_abs_peak_index(out, audio_1_sample_count - audio_2_sample_count);
	free(out);
	return index;
}

void compute_derivative(float *audio, size_t sample_count) {
	float last_sample = audio[0];
	for (int i = 1; i < sample_count; i++) {
		float sample = audio[i];
		audio[i] = sample - last_sample;
		last_sample = sample;
	}
}

size_t find_offset_correlation_on_derivatives(float *audio_1, size_t audio_1_sample_count, float *audio_2, size_t audio_2_sample_count) {
	float *out = malloc(audio_1_sample_count * sizeof(float));
	compute_derivative(audio_1, audio_1_sample_count);
	compute_derivative(audio_2, audio_2_sample_count);

#pragma omp parallel for
	for (int i = 0; i < audio_1_sample_count - audio_2_sample_count; i++) {
		float sum = 0;
		for (int j = 0; j < audio_2_sample_count; j++) {
			sum += audio_1[i+j] * audio_2[j];
		}
		out[i] = sum;
	}

	size_t index = find_abs_peak_index(out, audio_1_sample_count - audio_2_sample_count);
	free(out);
	return index;
}

size_t find_offset_in_files(char *path_1, char *path_2, size_t (*algorithm)(float*, size_t, float*, size_t)) {
	//printf("find offset %s and %s\n", path_1, path_2);
	fflush(stdout);

	int fd;
	struct stat st;

	// Read path_1
	fd = open(path_1, O_RDONLY);
	assert(fstat(fd, &st) != -1);
	size_t file_1_size = st.st_size;
	float *file_1_data = malloc(file_1_size);
	int _ = read(fd, file_1_data, file_1_size);

	// Read path_2
	fd = open(path_2, O_RDONLY);
	assert(fstat(fd, &st) != -1);
	size_t file_2_size = st.st_size;
	float *file_2_data = malloc(file_2_size);
	_ = read(fd, file_2_data, file_2_size);

	return algorithm(file_1_data, file_1_size / sizeof(float), file_2_data, file_2_size / sizeof(float));
}

int main() {
	//find_offset_in_files("tests/orig_piano_pc/1.raw", "tests/orig_piano_pc/2.raw");
	//find_offset_in_files("tests/orig_piano_pc/2.raw", "tests/orig_piano_pc/1.raw");
	//return 0;

	Algorithm algos[] = {temporary, find_offset_correlation, find_offset_correlation_hpf, find_offset_correlation_volume_normalized, find_offset_correlation_on_derivatives, find_offset_zero_crossings};
	int algos_len = sizeof(algos) / sizeof(algos[0]);

	char *algo_names[] = {"temporary", "Correlation", "Correlation (HPF @ 20hz)", "Correlation (volume normalized)", "Correlation on derivatives", "Zero-crossings"};

	for (int algo_idx = 0; algo_idx < algos_len; algo_idx++) {
		fprintf(stderr, "%s:\n", algo_names[algo_idx]);

		DIR *dir;
		struct dirent *entry;
		char child[PATH_MAX];

		if (!(dir = opendir("tests"))) {
			return 1;
		}

		// Iterate "tests" directory
		while ((entry = readdir(dir)) != NULL) {
			if (entry->d_name[0] == '.') {
				continue;
			}

			// Read "expected_offset_seconds.txt"
			snprintf(child, PATH_MAX, "tests/%s/expected_offset_seconds.txt", entry->d_name);
			int fd = open(child, O_RDONLY);

			struct stat st;
			if (fstat(fd, &st) == -1) {
				return 2;
			}
			char *expectedOffsetStr = malloc(st.st_size);
			int _ = read(fd, expectedOffsetStr, st.st_size);
			expectedOffsetStr[st.st_size-1] = 0; // Remove trailing newline by replacing it with a null byte.
			close(fd);

			char path_1[PATH_MAX];
			char path_2[PATH_MAX];
			snprintf(path_1, PATH_MAX, "tests/%s/1.raw", entry->d_name);
			snprintf(path_2, PATH_MAX, "tests/%s/2.raw", entry->d_name);

			if (memcmp(path_1, "tests/temp/1.raw", 16) != 0) continue;

			size_t index = find_offset_in_files(path_1, path_2, algos[algo_idx]);
			float offset = (float)index / 48000;
			fprintf(stderr, "Test \x1b[0;32m|%20s|\x1b[0m expected: %s, got: %f\n", entry->d_name, expectedOffsetStr, offset);

			free(expectedOffsetStr);
			break; // DEBUGGING
		}
		break; // DEBUGGING
	}
}
