#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GPU_CLK 0.3
#define GPU_CORES 384
#define GPU_MOD 2

//#define M 1024
//#define N 1024
//#define K 1024

#define TS 16

#define NUM_REPETITION 100

//#define A_size M * K
//#define B_size N * K
//#define C_size M * N

int M = 1024;
int N = 1024;
int K = 1024;

int A_size;
int B_size;
int C_size;

void initMatrix(float* mat, int size);
void printMatrix(float* mat, int size);
void printDeviceInfo(cl_device_id device);
char* readKernelFile(const char* fileName, long* fileSize);
void verifyMatrix(float* A, float* B, float* matrix);

int main(int argc, char* argv[]) {

	M = strtol(argv[1], NULL, 10);
	N = strtol(argv[2], NULL, 10);
	K = strtol(argv[3], NULL, 10);

	A_size = M * K;
	B_size = N * K;
	C_size = M * N;

	// Init matrices A and B
	srand(1000);
	float* A = (float*)malloc(A_size * sizeof(float));
	initMatrix(A, A_size);
	float* B = (float*)malloc(B_size * sizeof(float));
	initMatrix(B, B_size);

	// Allocate memory for matrix C
	float* C = (float*)malloc(C_size * sizeof(float));


	// Initialize OpenCL
	cl_int err;

	// Get platform
	cl_platform_id platform;
	err = clGetPlatformIDs(1, &platform, NULL);

	// Get the device
	cl_device_id device;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	// Create a context
	cl_context context;
	context = clCreateContext(0, 1, &device, NULL, NULL, &err);

	// Create command queue
	cl_command_queue command_queue;
	command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

	// Print device info
	printDeviceInfo(device);

	// Read kernel file
	char* source;
	long sourceSize;
	source = readKernelFile("matrix_mul_tiles.cl", &sourceSize);

	// Create program
	cl_program program;
	program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);

	// Build program
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create kernel
	cl_kernel kernel;
	kernel = clCreateKernel(program, "matrix_mul", &err);

	// Create memory objects
	cl_mem A_buff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * A_size, A, &err);
	cl_mem B_buff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * B_size, B, &err);
	cl_mem C_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * C_size, NULL, &err);

	// Set arguments
	int m = M; int n = N; int k = K;
	int num_of_works = pow(K / TS + (K % TS == 0 ? 0 : 1), 2);
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (const void*)&A_buff);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void*)&B_buff);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void*)&C_buff);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (const void*)&m);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (const void*)&n);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (const void*)&k);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (const void*)&num_of_works);

	// Configure the work-group
	size_t local[2] = { TS, TS };
	size_t global[2] = { 48, 32 };

	printf("num_of_works = %d\n", num_of_works);
	printf("local = { %d, %d }\n", (int)local[0], (int)local[1]);
	printf("global = { %d, %d }\n", (int)global[0], (int)global[1]);

	// Time event
	cl_event time_event[NUM_REPETITION];

	// Run kernel
	for (int i = 0; i < NUM_REPETITION; i++) {
		err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, local, 0, NULL, &time_event[i]);
		if (err != CL_SUCCESS)
			printf("err = %d\n", err);
	}


	// Print time info
	clFinish(command_queue);
	cl_ulong time_start[NUM_REPETITION], time_end[NUM_REPETITION], exec_time[NUM_REPETITION];
	for (int i = 0; i < NUM_REPETITION; i++) {
		err = clGetEventProfilingInfo(time_event[i], CL_PROFILING_COMMAND_START, sizeof(time_start[i]), &time_start[i], NULL);
		err = clGetEventProfilingInfo(time_event[i], CL_PROFILING_COMMAND_END, sizeof(time_end[i]), &time_end[i], NULL);
		exec_time[i] = time_end[i] - time_start[i];
	}
	// average time
	cl_ulong average = (time_end[NUM_REPETITION - 1] - time_start[0]) / NUM_REPETITION;

	// stddev
	cl_ulong sum = 0;
	for (int i = 0; i < NUM_REPETITION; i++)
		sum = sum + pow(abs(exec_time[i] - average), 2);
	
	cl_ulong std_deviation = sqrt(sum / (float)NUM_REPETITION);
	// Print average time and stddev
	printf("Average execution time = %.3lf ms.\n", (double)average / 1000000);
	printf("Standard deviation = %.3lf ms.\n", (double)std_deviation / 1000000);

	// GFLOPS
	double peak = GPU_CLK * GPU_CORES * GPU_MOD;
	double gflops = ((double)((double)K * (double)M * (double)N * 2) / 1000000000) / ((double)average / 1000000000);
	printf("Efficiency = %.2lf%%\n", gflops * 100 / peak);

	// Get result
	err = clEnqueueReadBuffer(command_queue, C_buff, CL_TRUE, 0, C_size * sizeof(*C), C, 0, NULL, NULL);


	// Verify
	verifyMatrix(A, B, C);

	//Cleanup
	free(A);
	free(B);
	free(C);

	// Free memory objects
	clReleaseMemObject(A_buff);
	clReleaseMemObject(B_buff);
	clReleaseMemObject(C_buff);

	// Release
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	clReleaseProgram(program);
	clReleaseKernel(kernel);

	return 0;
}

void initMatrix(float* mat, int size) {
	for (int i = 0; i < size; i++)
		mat[i] = (float)rand() / (float)RAND_MAX;
	// for (int i = 0; i < size; i++)
	//	 mat[i] = 1;
}

void printMatrix(float* mat, int size) {

	printf("\n\nMatrix\n");
	for (int i = 0; i < size; i++)
	{
		if ((i % M) == 0 && i != 0) printf("\n");
		printf("%.0lf ", mat[i]);
	}
	printf("\n");
}

void printDeviceInfo(cl_device_id device) {
	// Print device name
	size_t deviceNameSize;
	char deviceName[1024];
	clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &deviceNameSize);
	clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName, NULL);
	printf("Device name: %s\n", deviceName);

	// Print number of compute units
	cl_uint maxComputeUnits;
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
	printf("Number of compute units: %d\n", maxComputeUnits);

	// Print max work group size
	size_t maxWorkGroupSize;
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
	printf("Max work group size: %zu\n", maxWorkGroupSize);

	// Print global mem size
	cl_ulong globalMemSize;
	clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);
	printf("Global memory size: %llu\n", globalMemSize);

	// Print local mem size
	cl_ulong localMemSize;
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
	printf("Local memory size: %llu\n", localMemSize);
}

char* readKernelFile(const char* fileName, long* fileSize) {
	FILE* file;
	fopen_s(&file, fileName, "rb");
	if (!file) {
		printf("File %s failed to open.\n", fileName);
		exit(1);
	}

	fseek(file, 0, SEEK_END);
	long size = ftell(file);
	rewind(file);

	char* source = (char*)malloc(sizeof(char) * ((long long)size + 1));
	if (!source) {
		printf("Memory allocation failed.\n");
		exit(2);
	}
	fread(source, 1, (long long)size * sizeof(char), file);
	source[size] = '\0';
	fclose(file);

	*fileSize = (size + 1);
	return source;
}

void verifyMatrix(float* A, float* B, float* matrix) {
	float temp;

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < K; j++) {
			float sum = 0.0;
			for (int k = 0; k < K; k++)
			{
				sum += A[i * M + k] * B[k * K + j];
			}
			temp = sum;

			if (temp != matrix[i * K + j]){
				printf("Error: Result is not correct!\n");
				return;
			}
		}
	}
	printf("Result is correct!");
}