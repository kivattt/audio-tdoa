// In the tests folder, 2.raw is always shorter than 1.raw
// We assume all audio files are 32 bit floating point raw data

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <dirent.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))

float find_offset_2(float *audio_1, size_t audio_1_samplecount, float *audio_2, size_t audio_2_samplecount) {
	size_t buf_size = audio_2_samplecount * sizeof(float);

	float *current_row = malloc(buf_size);
	float *prev_row = malloc(buf_size);
	float *prev_prev_row = malloc(buf_size);

	memset(prev_prev_row, 0, buf_size);
	for (int i = 0; i < buf_size / sizeof(float); i++) {
		prev_row[i] = INFINITY;
	}
	memset(current_row, 0, buf_size);

	/**/



	/**/

	free(current_row);
	free(prev_row);
	free(prev_prev_row);
}

float find_offset(float *audio_1, size_t audio_1_samplecount, float *audio_2, size_t audio_2_samplecount) {
	float *out = malloc(audio_1_samplecount * sizeof(float));

	float *window = malloc(audio_2_samplecount * sizeof(float));
	for (int i = 0; i < audio_2_samplecount; i++) {
		float x = 2.0 * i / audio_2_samplecount - 1.0;
		window[i] = pow(M_E, -6 * x*x);
	}

//#pragma omp parallel for schedule(dynamic)
#pragma omp parallel for
	for (int i = 0; i < audio_1_samplecount - audio_2_samplecount; i++) {
		float sum = 0;
		for (int j = 0; j < audio_2_samplecount; j++) {
			sum += window[j] * audio_1[i+j] * audio_2[j];
		}
		out[i + audio_2_samplecount / 2] = sum / 10.06;
	}

	for (int i = 0; i < audio_1_samplecount; i++) {
		write(1, &out[i], 4);
	}
	fflush(stdout);

	free(window);
	free(out);
	return 0.0;
}

float find_offset_in_files(char *path_1, char *path_2) {
	//printf("find offset %s and %s\n", path_1, path_2);
	fflush(stdout);

	int fd;
	struct stat st;

	// Read path_1
	fd = open(path_1, O_RDONLY);
	assert(fstat(fd, &st) != -1);
	size_t file_1_size = st.st_size;
	float *file_1_data = malloc(file_1_size);
	read(fd, file_1_data, file_1_size);

	// Read path_2
	fd = open(path_2, O_RDONLY);
	assert(fstat(fd, &st) != -1);
	size_t file_2_size = st.st_size;
	float *file_2_data = malloc(file_2_size);
	read(fd, file_2_data, file_2_size);

	return find_offset(file_1_data, file_1_size / sizeof(float), file_2_data, file_2_size / sizeof(float));
}

int main() {
	//find_offset_in_files("tests/orig_piano_pc/1.raw", "tests/orig_piano_pc/2.raw");
	find_offset_in_files("tests/orig_piano_pc/1.raw", "verb.raw");
	return 0;

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
		read(fd, expectedOffsetStr, st.st_size);
		expectedOffsetStr[st.st_size-1] = 0; // Remove trailing newline by replacing it with a null byte.
		close(fd);

		char path_1[PATH_MAX];
		char path_2[PATH_MAX];
		snprintf(path_1, PATH_MAX, "tests/%s/1.raw", entry->d_name);
		snprintf(path_2, PATH_MAX, "tests/%s/2.raw", entry->d_name);

		//printf("Test \x1b[0;32m%s\x1b[0m expected: %s, got: %s\n", entry->d_name, expectedOffsetStr, "TEMP");
		find_offset_in_files(path_1, path_2);

		free(expectedOffsetStr);
	}
}
