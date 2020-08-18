// Includes for the project
#include <stdio.h>
#include <cmath>
#include <iomanip>
#include <boost/lexical_cast.hpp>
#include <CL/cl.h>
#include <math.h>
#include <stdlib.h>
#include <fftw3.h>
#include <complex>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <Core/Time.hpp>
#include <OpenCL/Event.hpp>
#include <Core/Assert.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>
#include "pgm.h"

#define PI 3.14159265358979
#define MAX_SOURCE_SIZE (0x100000)
#define MODULUS(a, b) (sqrt((a) * (a) + (b) * (b)))

using namespace cv;
using namespace std;

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////

//Shift the zero frequency component to the center of the spectrum.
void ShiftFFT(Mat magnitudeImage) {

	// crop the image if it has an odd number of rows or columns
	magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));

	int cols = magnitudeImage.cols / 2;
	int rows = magnitudeImage.rows / 2;

	//Quadrant formation of Spectrum
	Mat a0(magnitudeImage, Rect(0, 0, cols, rows));
	Mat a1(magnitudeImage, Rect(cols, 0, cols, rows));
	Mat a2(magnitudeImage, Rect(0, rows, cols, rows));
	Mat a3(magnitudeImage, Rect(cols, rows, cols, rows));

	//To swap the quadrants
	Mat temp;
	a0.copyTo(temp);
	a3.copyTo(a0);
	temp.copyTo(a3);
	a1.copyTo(temp);
	a2.copyTo(a1);
	temp.copyTo(a2);
}

//To calculate the magnitude of the FFT Spectrum
void magFFT(const Mat& complexVal, Mat& output) {

	vector<Mat> zeroplanes;
	split(complexVal, zeroplanes);
	magnitude(zeroplanes[0], zeroplanes[1], output);
}

//To compute the FFT of the input image
void computeFFT(Mat& image, Mat& output) {

	Mat padded;

	int m = getOptimalDFTSize(image.rows);
	int n = getOptimalDFTSize(image.cols);

	// Add zero values at the border
	copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols,
			BORDER_REPLICATE);

	Mat padded_img;
	padded.convertTo(padded_img, CV_32F);
	dft(padded_img, output, DFT_COMPLEX_OUTPUT);

}

//To display the FFT Spectrum
void DisplayFFTSpectrum(string wname, const Mat& complex) {

	Mat magnitudeImage;

	magFFT(complex, magnitudeImage);

	//switch to logarithmic scale
	log(magnitudeImage + 1.0, magnitudeImage);

	ShiftFFT(magnitudeImage);

	normalize(magnitudeImage, magnitudeImage, 1, 0, NORM_INF);

	cv::imshow(wname, magnitudeImage);

}

//////////////////////////////////////////////////////////////////////////////
// GPU implementation
//////////////////////////////////////////////////////////////////////////////

/*Our Implementation of 2D FFT*/
cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue queue = NULL;
cl_program program = NULL;

enum Mode {
	forward = 0, inverse = 1
};

//To set the global work item size and local work item size
int setWorkSize(size_t *GlobalWorkSize, size_t *localWorkSize, cl_int x, cl_int y) {
	switch (y) {
	case 1:
		GlobalWorkSize[0] = x;
		GlobalWorkSize[1] = 1;
		localWorkSize[0] = 4;
		localWorkSize[1] = 4;
		break;
	default:
		GlobalWorkSize[0] = x;
		GlobalWorkSize[1] = y;
		localWorkSize[0] = 16;
		localWorkSize[1] = 16;
		break;
	}

	return 0;
}

//Kernel calls for FFT Operation
Core::TimeSpan CoreFFTImpl(cl_mem dest, cl_mem source, cl_mem spin, cl_int m,
		enum Mode direction) {

	cl_int ret;

	cl_int iteration;
	cl_uint flag;

	cl_int n = 1 << m;

	cl_event kernelDone;
	cl_event BitReverseEvent;
	cl_kernel bitReverse = NULL;
	cl_kernel butterfly = NULL;
	cl_kernel normalizeSample = NULL;

	//Create the kernels
	bitReverse = clCreateKernel(program, "reverseBit", &ret);
	butterfly = clCreateKernel(program, "butterflyOperation", &ret);
	normalizeSample = clCreateKernel(program, "normalizeSamples", &ret);

	size_t GlobalWorkSize[2];
	size_t localWorkSize[2];

	switch (direction) {
	case forward:
		flag = 0x00000000;
		break;
	case inverse:
		flag = 0x80000000;
		break;
	}

	//Set the arguments of kernels
	clSetKernelArg(bitReverse, 0, sizeof(cl_mem), (void *) &dest);
	clSetKernelArg(bitReverse, 1, sizeof(cl_mem), (void *) &source);
	clSetKernelArg(bitReverse, 2, sizeof(cl_int), (void *) &m);
	clSetKernelArg(bitReverse, 3, sizeof(cl_int), (void *) &n);

	clSetKernelArg(butterfly, 0, sizeof(cl_mem), (void *) &dest);
	clSetKernelArg(butterfly, 1, sizeof(cl_mem), (void *) &spin);
	clSetKernelArg(butterfly, 2, sizeof(cl_int), (void *) &m);
	clSetKernelArg(butterfly, 3, sizeof(cl_int), (void *) &n);
	clSetKernelArg(butterfly, 5, sizeof(cl_uint), (void *) &flag);

	clSetKernelArg(normalizeSample, 0, sizeof(cl_mem), (void *) &dest);
	clSetKernelArg(normalizeSample, 1, sizeof(cl_int), (void *) &n);

	//Reverse bit ordering
	setWorkSize(GlobalWorkSize, localWorkSize, n, n);
	clEnqueueNDRangeKernel(queue, bitReverse, 2, NULL, GlobalWorkSize, localWorkSize, 0, NULL,
			&BitReverseEvent);

	clWaitForEvents(1, &BitReverseEvent);

	//Perform Butterfly Operations
	Core::TimeSpan time_temp = OpenCL::getElapsedTime(BitReverseEvent);
	setWorkSize(GlobalWorkSize, localWorkSize, n / 2, n);

	for (iteration = 1; iteration <= m; iteration++) {

		clSetKernelArg(butterfly, 4, sizeof(cl_int), (void *) &iteration);
		clEnqueueNDRangeKernel(queue, butterfly, 2, NULL, GlobalWorkSize, localWorkSize, 0,
				NULL, &kernelDone);
		clWaitForEvents(1, &kernelDone);

		Core::TimeSpan time3 = OpenCL::getElapsedTime(kernelDone);
		time_temp = time3 + time_temp;

	}


	if (direction == inverse) {
		setWorkSize(GlobalWorkSize, localWorkSize, n, n);
		clEnqueueNDRangeKernel(queue, normalizeSample, 2, NULL, GlobalWorkSize, localWorkSize, 0,
				NULL, &kernelDone);
		clWaitForEvents(1, &kernelDone);
	}

	clReleaseKernel(butterfly);
	clReleaseKernel(bitReverse);
	clReleaseKernel(normalizeSample);

	return time_temp;
}


//Main Function
int main() {
	//////////////////////////////////////////////////////////////////////////////
	// Please change to your local workspace folder
	//////////////////////////////////////////////////////////////////////////////
	Mat input_img = imread("/home/raobhaah/workspace/Project_2d_FFT/lena.pgm", 0);
	const char ImagePath[] = "lena.pgm";
	const char fileName[] =	"/home/raobhaah/workspace/Project_2d_FFT/src/kernel.cl";
	//////////////////////////////////////////////////////////////////////////////

	std::cout << "/////////////////////////////////////////////////////////////////////////////////"<< std::endl;
	std::cout <<"// Fachpraktikum High Performance Programming with Graphic Cards               //"<< std::endl;
	std::cout <<"////////////////////////////////////////////////////////////////////////////// "<< "\n"<<std::endl;


	//Start CPU Time measurement
	Core::TimeSpan time1 = Core::getCurrentTime();

	cv::Mat imgIFFT;

	Mat FFTOutput;

	uchar *Source_data;
	uchar *imgIFFT_data;

	fftw_complex *input_data;
	fftw_complex *FFT;
	fftw_complex *IFFT;
	fftw_plan plan_FFT;
	fftw_plan plan_IFFT;

	int width, height, step;
	int a, b, k, r;

	using namespace cv;
	using namespace std;

	cl_mem xmobj = NULL;
	cl_mem rmobj = NULL;
	cl_mem wmobj = NULL;

	cl_kernel spinFac = NULL;
	cl_kernel transpose = NULL;

	cl_platform_id platform_id = NULL;

	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;

	cl_int ret;

	cl_float2 *tempdata_1;
	cl_float2 *tempdata_2;
	cl_float2 *tempdata_3;

	cl_event WriteBufferEvent;
	cl_event ReadBufferEvent1;
	cl_event ReadBufferEvent2;

	pgm_t inputImg_pgm;
	pgm_t outputpgm;

	FILE *fp;
	size_t source_size;
	char *source_str;

	cl_int i, j;
	cl_int n;
	cl_int m;

	size_t GlobalWorkSize[2];
	size_t localWorkSize[2];

	//////////////////////////////////////////////////////////////////////////////
	// Start of CPU implementation
	//////////////////////////////////////////////////////////////////////////////

	if (!input_img.data) {
		fprintf( stderr, "Cannot load file!\n");
		return 1;
	}

	//Create new image for IFFT result
	imgIFFT = input_img.clone();

	//Get image dimensions
	width = input_img.size().width;
	height = input_img.size().height;
	step = input_img.step;

	Source_data = (uchar*) input_img.data;
	imgIFFT_data = (uchar*) imgIFFT.data;

	//Initialize arrays for FFT operations using FFTW Library
	input_data = fftw_alloc_complex(width * height);
	FFT = fftw_alloc_complex(width * height);
	IFFT = fftw_alloc_complex(width * height);

	//Create plans for FFT and IFFT
	plan_FFT = fftw_plan_dft_1d(width * height, input_data, FFT, FFTW_FORWARD,
			FFTW_ESTIMATE);
	plan_IFFT = fftw_plan_dft_1d(width * height, FFT, IFFT, FFTW_BACKWARD,
			FFTW_ESTIMATE);

	//Load source data to input of FFT operations
	for (a = 0, k = 0; a < height; a++) {
		for (b = 0; b < width; b++) {
			input_data[k][0] = (double) Source_data[a * step + b];
			input_data[k][1] = 0.0;
			k++;
		}
	}

	//Perform FFT
	fftw_execute(plan_FFT);
	computeFFT(input_img, FFTOutput);

	//Perform IFFT
	fftw_execute(plan_IFFT);

	// normalize IFFT result
	for (a = 0; a < (width * height); a++) {
		IFFT[a][0] /= (double) (width * height);
	}

	// copy IFFT result to Output IFFT Image
	for (a = 0, k = 0; a < height; a++) {
		for (b = 0; b < width; b++) {
			imgIFFT_data[a * step + b] = (uchar) IFFT[k++][0];
		}
	}

	//Display spectrum
	DisplayFFTSpectrum("FFT_Spectrum_CPU", FFTOutput);

	//End Time
	Core::TimeSpan time2 = Core::getCurrentTime();
	Core::TimeSpan time = time2 - time1;

	std::cout << "CPU Execution completed. Images Displayed!" <<"\n"<< std::endl;
	//////////////////////////////////////////////////////////////////////////////
	// End of CPU implementation
	//////////////////////////////////////////////////////////////////////////////







	//////////////////////////////////////////////////////////////////////////////
	// Start of GPU implementation
	//////////////////////////////////////////////////////////////////////////////

	/* Load kernel source code */
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char *) malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp); //Storing kernel source code path in source_str
	fclose(fp);


	//Read input image
	r = readPGM(&inputImg_pgm, ImagePath);
	if (r < 0) {
		fprintf(stderr, "Wrong input image format. Exiting...\n");
		exit(1);
	}

	n = inputImg_pgm.width;
	m = (cl_int) (log((double) n) / log(2.0));

	tempdata_1 = (cl_float2 *) malloc(n * n * sizeof(cl_float2));
	tempdata_2 = (cl_float2 *) malloc(n * n * sizeof(cl_float2));
	tempdata_3 = (cl_float2 *) malloc(n / 2 * sizeof(cl_float2));

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			((float *) tempdata_1)[(2 * n * j) + 2 * i + 0] =
					(float) inputImg_pgm.buf[n * j + i];
			((float *) tempdata_1)[(2 * n * j) + 2 * i + 1] = (float) 0;
		}
	}

	/* Get platform/device  */
	clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,
			&ret_num_devices);


	// Create OpenCL context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	//* Create Command queue
	queue = clCreateCommandQueue(context, device_id,
			CL_QUEUE_PROFILING_ENABLE, &ret);

	//Create Buffer Objects
	xmobj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			n * n * sizeof(cl_float2), NULL, &ret);

	rmobj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			n * n * sizeof(cl_float2), NULL, &ret);

	wmobj = clCreateBuffer(context, CL_MEM_READ_WRITE,
			(n / 2) * sizeof(cl_float2), NULL, &ret);


	/* Transfer data to memory buffer */
	clEnqueueWriteBuffer(queue, xmobj, CL_TRUE, 0,
			n * n * sizeof(cl_float2), tempdata_1, 0, NULL, &WriteBufferEvent);

	Core::TimeSpan time4 = OpenCL::getElapsedTime(WriteBufferEvent);

	//Create kernel program from source
	program = clCreateProgramWithSource(context, 1, (const char **) &source_str,
			(const size_t *) &source_size, &ret);

	//Build kernel program
	clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	//Create OpenCL Kernel
	spinFac = clCreateKernel(program, "spinFactor", &ret);
	transpose = clCreateKernel(program, "transposeMatrix", &ret);

	//Create spin factor
	clSetKernelArg(spinFac, 0, sizeof(cl_mem), (void *) &wmobj);
	clSetKernelArg(spinFac, 1, sizeof(cl_int), (void *) &n);
	setWorkSize(GlobalWorkSize, localWorkSize, n / 2, 1);
	clEnqueueNDRangeKernel(queue, spinFac, 1, NULL, GlobalWorkSize, localWorkSize, 0, NULL,
			NULL);

	std::cout << "Butterfly Operation 1 : FFT on rows" << ":" << std::endl;

	//Butterfly Operation
	Core::TimeSpan k1 = CoreFFTImpl(rmobj, xmobj, wmobj, m, forward);

	std::cout << "Cumulative GPU Time after Butterfly Operation 1 :" << k1
			<< "\n" << std::endl;

	//Transpose matrix
	clSetKernelArg(transpose, 0, sizeof(cl_mem), (void *) &xmobj);
	clSetKernelArg(transpose, 1, sizeof(cl_mem), (void *) &rmobj);
	clSetKernelArg(transpose, 2, sizeof(cl_int), (void *) &n);
	setWorkSize(GlobalWorkSize, localWorkSize, n, n);
	clEnqueueNDRangeKernel(queue, transpose, 2, NULL, GlobalWorkSize, localWorkSize, 0, NULL,
			NULL);

	std::cout << "Butterfly Operation 2 : FFT on columns" << ":" << std::endl;
	//Butterfly Operationcv::resizeWindow("FFT_Spectrum_CPU", 300,300);
	Core::TimeSpan k2 = CoreFFTImpl(rmobj, xmobj, wmobj, m, forward);
	std::cout << "Cumulative GPU Time after Butterfly Operation 2 :" << k2
			<< "\n" << std::endl;

	/* Read data from memory buffer */
	clEnqueueReadBuffer(queue, rmobj, CL_TRUE, 0,
			n * n * sizeof(cl_float2), tempdata_1, 0, NULL, &ReadBufferEvent1);

	Core::TimeSpan time5 = OpenCL::getElapsedTime(ReadBufferEvent1);

	float *fft_val;
	fft_val = (float *) malloc(n * n * sizeof(float));
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			fft_val[n * ((i)) + ((j))] = (MODULUS(((float * )tempdata_1)[(2 * n * i) + 2 * j],
					((float * )tempdata_1)[(2 * n * i) + 2 * j + 1]));
			fft_val[n * ((i)) + ((j))] = log(fft_val[n * ((i)) + ((j))]);
		}

	}

	outputpgm.width = n;
	outputpgm.height = n;
	normalizeF2PGM(&outputpgm, fft_val);
	free(fft_val);

	//Write temporary FFT Spectrum
	writePGM(&outputpgm, "FFT_Spectrum_GPU_temp.pgm");

	//Shift the FFT SPectrum to the center
	Mat img_c;
	img_c = cv::imread("FFT_Spectrum_GPU_temp.pgm");
	ShiftFFT(img_c);
	cv::imwrite("FFT_Spectrum_GPU.pgm", img_c);


	//Inverse FFT
	std::cout << "Butterfly Operation 3 : IFFT on rows" << ":" << std::endl;

	//Butterfly Operation
	Core::TimeSpan k3 = CoreFFTImpl(xmobj, rmobj, wmobj, m, inverse);
	std::cout << "Cumulative GPU Time after Butterfly Operation 3 :" << k3
			<< "\n" << std::endl;

	//Transpose matrix
	clSetKernelArg(transpose, 0, sizeof(cl_mem), (void *) &rmobj);
	clSetKernelArg(transpose, 1, sizeof(cl_mem), (void *) &xmobj);

	setWorkSize(GlobalWorkSize, localWorkSize, n, n);
	clEnqueueNDRangeKernel(queue, transpose, 2, NULL, GlobalWorkSize, localWorkSize, 0, NULL,
			NULL);


	std::cout << "Butterfly Operation 4 : IFFT on columns" << ":" << std::endl;
	//Butterfly Operation
	Core::TimeSpan k4 = CoreFFTImpl(xmobj, rmobj, wmobj, m, inverse);
	std::cout << "Cumulative GPU Time after Butterfly Operation 4 : " << k4
			<< "\n" << std::endl;

	//////////////////////////////////////////////////////////////////////////////
	//End of GPU implementation
	//////////////////////////////////////////////////////////////////////////////





	//////////////////////////////////////////////////////////////////////////////
	//Calculation of Execution Times and Speed up
	//////////////////////////////////////////////////////////////////////////////

	//Total Time on CPU
	std::cout << "Total Time on CPU : " << time  << std::endl;

	//Total Time on GPU
	Core::TimeSpan TotalGPUTime = k1 + k2 + k3 + k4;
	std::cout << "Total Time on GPU : " << TotalGPUTime <<"\n"<< std::endl;

	//Speed up without memory transactions
	double SpeedUp = (time.getSeconds() / TotalGPUTime.getSeconds());
	std::cout << "SpeedUp without memory transactions : " << SpeedUp << std::endl;

	/* Read data from memory buffer */
	clEnqueueReadBuffer(queue, xmobj, CL_TRUE, 0,
			n * n * sizeof(cl_float2), tempdata_1, 0, NULL, &ReadBufferEvent2);


	Core::TimeSpan time6 = OpenCL::getElapsedTime(ReadBufferEvent2);
	Core::TimeSpan MemTransacTime = time4+time5+time6;
	MemTransacTime = MemTransacTime+TotalGPUTime;

	//Speed up with memory transactions
	double SpeedUp_Mem = (time.getSeconds() / MemTransacTime.getSeconds());
	std::cout << "SpeedUp with memory transactions : " << SpeedUp_Mem <<"\n"<< std::endl;
	float *ifft_val;
	ifft_val = (float *) malloc(n * n * sizeof(float));

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			ifft_val[n * ((i)) + ((j))] = (MODULUS(((float * )tempdata_1)[(2 * n * i) + 2 * j],
					((float * )tempdata_1)[(2 * n * i) + 2 * j + 1]));
		}
	}
	outputpgm.width = n;
	outputpgm.height = n;
	normalizeF2PGM(&outputpgm, ifft_val);
	free(ifft_val);

	// Write out image
	writePGM(&outputpgm, "IFFT_GPU.pgm");

	// Read the GPU IFFT Output
	Mat img_GPU = cv::imread("IFFT_GPU.pgm");

	std::cout << "GPU Execution completed. Images written to local workspace!" << std::endl;

	//Flush queue and release objects
	clFlush(queue);

	clFinish(queue);

	clReleaseKernel(transpose);

	clReleaseKernel(spinFac);

	clReleaseProgram(program);

	clReleaseMemObject(xmobj);

	clReleaseMemObject(rmobj);

	clReleaseMemObject(wmobj);

	clReleaseCommandQueue(queue);

	clReleaseContext(context);

	// display images from CPU
	cv::namedWindow("Input_Image", CV_WINDOW_AUTOSIZE);

	cv::namedWindow("IFFT_CPU", CV_WINDOW_AUTOSIZE);

	cv::namedWindow("IFFT_GPU", CV_WINDOW_AUTOSIZE);

	cv::namedWindow("FFT_Spectrum_GPU", CV_WINDOW_AUTOSIZE);


	cv::imshow("Input_Image", input_img);
	cv::imshow("IFFT_CPU", imgIFFT);
	cv::imshow("IFFT_GPU", img_GPU);
	cv::imshow("FFT_Spectrum_GPU", img_c);

	cv::imwrite("IFFT_CPU.pgm", imgIFFT);

	std::cout << "Press 'q' to terminate... " <<"\n"<< std::endl;

	char key;
	while (true) {
		key = cv::waitKey(0);
		if ('q' == key)
			break;
	}

	// free memory
	destroyPGM(&inputImg_pgm);
	destroyPGM(&outputpgm);

	free(source_str);
	free(tempdata_3);
	free(tempdata_2);
	free(tempdata_1);
	cv::destroyWindow("original_image");
	cv::destroyWindow("IFFT");
	input_img.release();
	imgIFFT.release();
	fftw_destroy_plan(plan_FFT);
	fftw_destroy_plan(plan_IFFT);
	fftw_free(input_data);
	fftw_free(FFT);
	fftw_free(IFFT);
	return 0;
}
