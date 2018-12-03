#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <chrono>
#include <omp.h>
#include "cmath"

using namespace std;
using namespace cv;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void kernel(unsigned char* input, unsigned char* search, unsigned char* output, int inputWidth, int inputHeight, int searchWidth, int searchHeight, int* posX, int* posY, unsigned long long int* minsad, int* sadloc, int inputStep, int searchStep) {
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex < inputWidth - searchWidth) && (yIndex < inputHeight - searchHeight)) {

		int index = yIndex * inputStep + xIndex;
		unsigned long long int SAD = 0;

		output[index] = 0;

		// check every pixel of actual image with the template image
		// loop through the template image
		for (int t_j = 0; t_j < searchWidth; t_j++) {
			for (int t_i = 0; t_i < searchHeight; t_i++) {

				int inputIndex = (t_j + yIndex) * inputStep + (t_i + xIndex);
				int searchIndex = t_j * searchStep + t_i;

				SAD += powf(fabsf(int(input[inputIndex]) - int(search[searchIndex])), 2);
			}
		}

		output[index] = SAD;

		__syncthreads();

		// save the best found position
		if (*minsad > SAD) {
			atomicExch(minsad, SAD);
			atomicExch(posX, xIndex);
			atomicExch(posY, yIndex);
			atomicExch(sadloc, SAD);
			//printf("SAD: %d \n", SAD);
		}
	}
}

void objReconGPU(const Mat& input, const Mat& search, int& posX, int& posY, int& sadlock) {

	Mat output(input.rows, input.cols, CV_32S);
	output = 0;

	unsigned long long int minsad = 9999999999999999999;
	unsigned long long int sad = 0;
	int xloc = 0, yloc = 0, sadloc = 0;

	// Set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	SAFE_CALL(cudaSetDevice(dev), "Error setting device");

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	size_t inputBytes = input.step * input.rows;
	size_t searchBytes = search.step * search.rows;
	size_t outputBytes = output.step * output.rows;

	unsigned char *d_input, *d_search;
	unsigned char *d_output;
	int *d_xloc, *d_yloc, *d_sadloc;
	unsigned long long int *d_sad, *d_minsad;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, inputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_search, searchBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, outputBytes), "CUDA Malloc Failed");

	SAFE_CALL(cudaMalloc<int>(&d_xloc, sizeof(int)), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<int>(&d_yloc, sizeof(int)), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<int>(&d_sadloc, sizeof(int)), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc((void**)&d_sad, sizeof(unsigned long long int)), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc((void**)&d_minsad, sizeof(unsigned long long int)), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_search, search.ptr(), searchBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output, output.ptr(), outputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	SAFE_CALL(cudaMemcpy(d_xloc, &xloc, sizeof(int), cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_yloc, &yloc, sizeof(int), cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_sadloc, &sadloc, sizeof(int), cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_sad, &sad, sizeof(unsigned long long int), cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_minsad, &minsad, sizeof(unsigned long long int), cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(32, 32);

	// Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	// const dim3 grid((input.cols) / block.x, (input.rows) / block.y);

	// Launch the color conversion kernel
	auto start_cpu = chrono::high_resolution_clock::now();
	kernel <<<grid, block >>> (d_input, d_search, d_output, input.cols, input.rows, search.cols, search.rows, d_xloc, d_yloc, d_minsad, d_sadloc, static_cast<int>(input.step), static_cast<int>(search.step));
	auto end_cpu = chrono::high_resolution_clock::now();

	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	printf("GPU elapsed %f ms\n", duration_ms.count());

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, inputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	SAFE_CALL(cudaMemcpy(&xloc, d_xloc, sizeof(int), cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(&yloc, d_yloc, sizeof(int), cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(&sadloc, d_sadloc, sizeof(int), cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_search), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_xloc), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_yloc), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_sadloc), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_sad), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_minsad), "CUDA Free Failed");

	/*int minLocX = 0;
	int minLocY = 0;
	auto tempVal = output.at<int>(0, 0);

	for (int i = 0; i < output.rows - search.rows; i++) {
		for (int j = 0; j < output.cols - search.cols; j++) {
			if (output.at<int>(i, j) < tempVal) {
				tempVal = output.at<int>(i, j);
				minLocX = j;
				minLocY = i;
				cout << tempVal << " " << minLocX << " " << minLocY << endl;
			}
		}
	}

	cout << output.at<int>(446, 1525) << " " << output.at<int>(1525, 446) << " " << endl;*/

	posX = xloc;
	posY = yloc;
	sadlock = sadloc;
}

void objReconOpenMP(const Mat& input, const Mat& search, int& posX, int& posY, int& sadlock) {
	unsigned long long int minSAD = 9999999999999999999;
	unsigned long long int SAD = 0;
	int xloc = 0, yloc = 0, sadloc = 0;
	int a_i, a_j, t_j, t_i;

	auto start_cpu = chrono::high_resolution_clock::now();
	for (a_i = 0; a_i < input.rows - search.rows; a_i++) {
		for (a_j = 0; a_j < input.cols - search.cols; a_j++) {

			SAD = 0;

			// check every pixel of actual image with the template image
			// loop through the template image
			#pragma omp parallel for private(t_j, t_i) reduction(+:SAD)
			for (t_j = 0; t_j < search.cols; t_j++) {
				for (t_i = 0; t_i < search.rows; t_i++) {
					SAD += pow(abs(int(input.at<uchar>(a_i + t_i, a_j + t_j)) - int(search.at<uchar>(t_i, t_j))), 2);
				}
			}

			// save the best found position
			if (minSAD > SAD) {
				minSAD = SAD;
				yloc = a_i;
				xloc = a_j;
				sadloc = SAD;
				cout << "SAD: " << SAD << endl;
			}
		}
	}
	auto end_cpu = chrono::high_resolution_clock::now();

	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	printf("CPU elapsed %f ms\n", duration_ms.count());

	posX = xloc;
	posY = yloc;
	sadlock = sadloc;
}

int main() {
	cout << "Template matching \n";
	Mat tImage, aImage, cpuImg, gpuImg;

	tImage = imread("waldo-search-3.jpg", 0);
	aImage = imread("waldo-full-3.jpg", 0);
	cpuImg = imread("waldo-full-3.jpg");
	gpuImg = imread("waldo-full-3.jpg");

	// performing template matching using SAD method
	int xlocCPU = 0, ylocCPU = 0, sadlocCPU = 0;
	int xlocGPU = 0, ylocGPU = 0, sadlocGPU = 0;

	objReconOpenMP(aImage, tImage, xlocCPU, ylocCPU, sadlocCPU);
	objReconGPU(aImage, tImage, xlocGPU, ylocGPU, sadlocGPU);

	cout << "CPU: " << xlocCPU << "," << ylocCPU << " SAD is " << sadlocCPU << endl;
	cout << "GPU: " << xlocGPU << "," << ylocGPU << " SAD is " << sadlocGPU << endl;

	// create a rectangle to demarkate the region
	rectangle(cpuImg, Point(xlocCPU - 5, ylocCPU - 5), Point(xlocCPU + tImage.cols + 5, ylocCPU + tImage.rows + 5), Scalar(0, 0, 255), 5);
	rectangle(gpuImg, Point(xlocGPU - 5, ylocGPU - 5), Point(xlocGPU + tImage.cols + 5, ylocGPU + tImage.rows + 5), Scalar(0, 0, 255), 5);

	imshow("Template image", tImage);
	namedWindow("CPU", WINDOW_NORMAL);
	namedWindow("GPU", WINDOW_NORMAL);
	imshow("CPU", cpuImg);
	imshow("GPU", gpuImg);
	waitKey();
	destroyAllWindows();

	return 0;
}
