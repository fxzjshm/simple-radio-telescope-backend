#ifndef _DSMATH_H_
#define _DSMATH_H_

// File date: 29 - 07 - 2008

// Double single functions based on DSFUN90 package:
// http://crd.lbl.gov/~dhbailey/mpdist/index.html
// Partially adapted from NVIDIA's CUDA Mandelbrot demo

// Set functions for agnostic arithmetic
__host__ void set(float2 &a, double b);
__host__ void set(float &a, double b);
__device__ __host__ void set(float2 &a, const float b);
__device__ __host__ void set(float &a, const float2 b);
__device__ __host__ void set(float2 &a, const float2 b);
__device__ __host__ void set(float &a, const float b);

// Arithmetic operators
__device__ float2 operator-(const float2 a);
__device__ float2 operator+(const float2 a, const float2 b);
__device__ float2 operator-(const float2 a, const float2 b);
__device__ float2 operator*(const float2 a, const float2 b);
__device__ float2 operator/(const float2 a, const float2 b);

// Note that the / operator potentially makes use of the FMAD
// instruction and may lose some accuracy as a result.

// This function sets the DS number A equal to the double precision floating point number B. 
__host__ inline void set(float2 &a, double b) {
	a.x = (float)b;
	a.y = (float)(b - a.x);
}

// Store a (truncated) double in a float
__host__ inline void set(float &a, double b) {
	a = (float)b;
}

// Store a float into a double single
__device__ __host__ inline void set(float2 &a, const float b) {
	a.x = b;
	a.y = 0;
}

// Store the hi word of the double single in a float
__device__ __host__ inline void set(float &a, const float2 b) {
	a = b.x;
}

// Double single assignment
__device__ __host__ inline void set(float2 &a, const float2 b) {
	a = b;
}

// Float assignment
__device__ __host__ inline void set(float &a, const float b) {
	a = b;
}

// This function computes b = -a.
__device__ inline float2 operator-(const float2 a) {
	float2 b;
	b.x = -a.x;
	b.y = -a.y;

	return b;
}

// Based on dsadd from DSFUN90, analysis by Norbert Juffa from NVIDIA.
// For a and b of opposite sign whose magnitude is within a factor of two
// of each other either variant below loses accuracy. Otherwise the result
// is within 1.5 ulps of the correctly rounded result with 48-bit mantissa.
// This function computes c = a + b.
__device__ inline float2 operator+(const float2 a, const float2 b) {
	float2 c;
#if defined(__DEVICE_EMULATION__)
	volatile float t1, e, t2;
#else
	float t1, e, t2;
#endif

	// Compute dsa + dsb using Knuth's trick.
	t1 = a.x + b.x;
	e = t1 - a.x;
	t2 = ((b.x - e) + (a.x - (t1 - e))) + a.y + b.y;

	// The result is t1 + t2, after normalization.
	c.x = e = t1 + t2;
	c.y = t2 - (e - t1);

	return c;
}

// Based on dssub from DSFUN90
// This function computes c = a - b.
__device__ inline float2 operator-(const float2 a, const float2 b) {
	float2 c;
#if defined(__DEVICE_EMULATION__)
	volatile float t1, e, t2;
#else
	float t1, e, t2;
#endif

	// Compute dsa - dsb using Knuth's trick.
	t1 = a.x - b.x;
	e = t1 - a.x;
	t2 = ((-b.x - e) + (a.x - (t1 - e))) + a.y - b.y;

	// The result is t1 + t2, after normalization.
	c.x = e = t1 + t2;
	c.y = t2 - (e - t1);

	return c;
}

// This function multiplies DS numbers A and B to yield the DS product C.
// Based on: Guillaume Da Graça, David Defour. Implementation of Float-Float
// Operators on Graphics Hardware. RNC'7, pp. 23-32, 2006.
__device__ inline float2 operator*(const float2 a, const float2 b) {
	float2 c;
#if defined(__DEVICE_EMULATION__)
	volatile float up, vp, u1, u2, v1, v2, mh, ml;
	volatile uint32_t tmp;
#else
	float up, vp, u1, u2, v1, v2, mh, ml;
	uint32_t tmp;
#endif
	// This splits a.x and b.x into high-order and low-order words.
	tmp = (*(uint32_t *)&a.x) & ~0xFFF;	// Bit-style splitting from Reimar
	u1 = *(float *)&tmp;
	//up  = a.x * 4097.0f;
	//u1  = (a.x - up) + up;
	u2  = a.x - u1;
	tmp = (*(uint32_t *)&b.x) & ~0xFFF;
	v1 = *(float *)&tmp;
	//vp  = b.x * 4097.0f;
	//v1  = (b.x - vp) + vp;
	v2  = b.x - v1;

	// Multilply a.x * b.x using Dekker's method.
	mh  = __fmul_rn(a.x, b.x);
	ml  = (((__fmul_rn(u1, v1) - mh) + __fmul_rn(u1, v2)) + __fmul_rn(u2, v1)) + __fmul_rn(u2, v2);

	// Compute a.x * b.y + a.y * b.x
	ml  = (__fmul_rn(a.x, b.y) + __fmul_rn(a.y, b.x)) + ml;

	// The result is mh + ml, after normalization.
	c.x = up = mh + ml;
	c.y = (mh - up) + ml;
	return c;
}

// Based on dsdiv from DSFUN90.
// This function divides the DS number A by the DS number B to yield the DS
// quotient DSC.
__device__ inline float2 operator/(const float2 a, const float2 b) {
	float2 c;
#if defined(__DEVICE_EMULATION__)
	volatile float s1, cona, conb, a1, b1, a2, b2, c11, c21;
	volatile float c2, t1, e, t2, t12, t22, t11, t21, s2;
#else
	float s1, cona, conb, a1, b1, a2, b2, c11, c21;
	float c2, t1, e, t2, t12, t22, t11, t21, s2;
#endif 
	// Compute a DP approximation to the quotient.
	s1 = a.x / b.x;

	// This splits s1 and b.x into high-order and low-order words.
	cona = __fmul_rn(s1, 8193.0f);
	conb = __fmul_rn(b.x, 8193.0f);
	a1 = cona - (cona - s1);
	b1 = conb - (conb - b.x);
	a2 = s1 - a1;
	b2 = b.x - b1;

	// Multiply s1 * dsb(1) using Dekker's method.
	c11 = __fmul_rn(s1, b.x);
	c21 = (((__fmul_rn(a1, b1) - c11) + __fmul_rn(a1, b2)) + __fmul_rn(a2, b1)) + __fmul_rn(a2, b2);

	// Compute s1 * b.y (only high-order word is needed).
	c2 = s1 * b.y;

	// Compute (c11, c21) + c2 using Knuth's trick.
	t1 = c11 + c2;
	e = t1 - c11;
	t2 = ((c2 - e) + (c11 - (t1 - e))) + c21;

	// The result is t1 + t2, after normalization.
	t12 = t1 + t2;
	t22 = t2 - (t12 - t1);

	// Compute dsa - (t12, t22) using Knuth's trick.

	t11 = a.x - t12;
	e = t11 - a.x;
	t21 = ((-t12 - e) + (a.x - (t11 - e))) + a.y - t22;

	// Compute high-order word of (t11, t21) and divide by b.x.
	s2 = (t11 + t21) / b.x;

	// The result is s1 + s2, after normalization.
	c.x = s1 + s2;
	c.y = s2 - (c.x - s1);

	return c;
}

#endif // _DSMATH_H_

