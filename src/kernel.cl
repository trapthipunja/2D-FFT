#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

#define PI 3.14159265358979323846
#define PI_by_2 1.57079632679489661923

//Kernel to perform butterfly operations
__kernel void butterflyOperation(__global float2 *x, __global float2 *w, int i, int j,
		int iter, uint flag) {
	unsigned int global_id_1 = get_global_id(0);
	unsigned int global_id_2 = get_global_id(1);

	int bflySize = 1 << (iter - 1);
	int bflyGroupDist = 1 << iter;
	int bflyGroupNum = j >> iter;
	int bflyGroupBase = (global_id_1 >> (iter - 1)) * (bflyGroupDist);
	int bflyGroupOffset = global_id_1 & (bflySize - 1);

	int m = global_id_2 * j + bflyGroupBase + bflyGroupOffset;
	int n = m + bflySize;

	int l = bflyGroupNum * bflyGroupOffset;

	float2 x_a, x_b, xb_xx, xb_yy, wab, wayx, wbyx, result_m, result_n;

	x_a = x[m];
	x_b = x[n];
	xb_xx = x_b.xx;
	xb_yy = x_b.yy;

	wab = as_float2(as_uint2(w[l]) ^ (uint2)(0x0, flag));
	wayx = as_float2(as_uint2(wab.yx) ^ (uint2)(0x80000000, 0x0));
	wbyx = as_float2(as_uint2(wab.yx) ^ (uint2)(0x0, 0x80000000));

	result_m = x_a + xb_xx * wab + xb_yy * wayx;
	result_n = x_a - xb_xx * wab + xb_yy * wbyx;

	x[m] = result_m;
	x[n] = result_n;
}

//Matrix transpose implementation
__kernel void transposeMatrix(__global float2 *dst, __global float2 *src, int n) {
	unsigned int global_id_1 = get_global_id(0);
	unsigned int global_id_2 = get_global_id(1);

	unsigned int transp_1 = global_id_2 * n + global_id_1;
	unsigned int transp_2 = global_id_1 * n + global_id_2;

	dst[transp_2] = src[transp_1];
}

//Normalizing by the number of samples
__kernel void normalizeSamples(__global float2 *y, int m) {
	unsigned int global_id_1 = get_global_id(0);
	unsigned int global_id_2 = get_global_id(1);

	y[global_id_2 * m + global_id_1] = y[global_id_2 * m + global_id_1] / (float2)((float)m, (float)m);
}

//Pre compute the value of spin factor
__kernel void spinFactor(__global float2 *w, int p) {
	unsigned int k = get_global_id(0);

	float2 angle =
			(float2)(2 * k * PI / (float)p, (2 * k * PI / (float)p) + PI_by_2);
	w[k] = cos(angle);
}

//Reordering the input data in order of bit reversed index
__kernel void reverseBit(__global float2 *dest, __global float2 *source, int r,
		int s) {
	unsigned int global_id_1 = get_global_id(0);
	unsigned int global_id_2 = get_global_id(1);

	unsigned int p = global_id_1;
	p = (p & 0x55555555) << 1 | (p & 0xAAAAAAAA) >> 1;
	p = (p & 0x33333333) << 2 | (p & 0xCCCCCCCC) >> 2;
	p = (p & 0x0F0F0F0F) << 4 | (p & 0xF0F0F0F0) >> 4;
	p = (p & 0x00FF00FF) << 8 | (p & 0xFF00FF00) >> 8;
	p = (p & 0x0000FFFF) << 16 | (p & 0xFFFF0000) >> 16;

	p >>= (32 - r);

	dest[global_id_2 * s + p] = source[global_id_2 * s + global_id_1];
}
