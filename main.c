#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <dirent.h>

#define CHILD_SIZE 1024

int main() {
	DIR *dir;
	struct dirent *entry;
	char child[CHILD_SIZE];

	if (!(dir = opendir("tests"))) {
		return 1;
	}

	// Iterate "tests" directory
	while ((entry = readdir(dir)) != NULL) {
		if (entry->d_name[0] == '.') {
			continue;
		}

		// Read "expected_offset_seconds.txt"
		snprintf(child, CHILD_SIZE, "tests/%s/expected_offset_seconds.txt", entry->d_name);
		int fd = open(child, O_RDONLY);

		struct stat st;
		if (fstat(fd, &st) == -1) {
			return 2;
		}
		char *expectedOffsetStr = malloc(st.st_size);
		read(fd, expectedOffsetStr, st.st_size);
		expectedOffsetStr[st.st_size-1] = 0; // Remove trailing newline by replacing it with a null byte.
		close(fd);

		// Read 1.wav
		snprintf(child, CHILD_SIZE, "tests/%s/1.wav", entry->d_name);

		printf("Test \x1b[0;32m%s\x1b[0m expected: %s, got: %s\n", entry->d_name, expectedOffsetStr, "TEMP");

		free(expectedOffsetStr);
	}
}
