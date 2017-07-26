#ifndef CUDA_MATH_EXT_H
#define CUDA_MATH_EXT_H

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <helper_math.h> 

#define CUGLSL_PI 3.14159265359f

////GLSL Compatibility////
typedef unsigned int   uint;
typedef unsigned char  uchar;
typedef unsigned short ushort;

#define barrier() ( __syncthreads() )
//#define barrier() ( __threadfence(), __syncthreads() )
//#define barrier()

#define gl_SMIDNV glSMIDNV()

#define gl_LocalInvocationIndex (threadIdx.x+threadIdx.y*blockDim.x)
#define gl_WorkGroupID blockIdx
#define gl_NumWorkGroups blockDim

#define gl_ThreadInWarpNV glThreadInWarpNV()

__device__ uint glSMIDNV() { //__noinline__ 
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
}

__device__ uint glThreadInWarpNV() { //__noinline__ 
    uint ret;
    asm("mov.u32 %0, %laneid;" : "=r"(ret) );
    return ret;
}

template<class T>
__device__ T shuffleIndexedNV(T var, int srcLane, int width){
	return __shfl(var, srcLane, width);
}



template<class T>
__device__ T shuffleXorNV(T var, int laneMask, int width){
	return __shfl_xor(var, laneMask, width);
}

template<class T>
__device__ T shuffleDownNV(T var, int srcLane, int width){
	return __shfl_down(var, srcLane, width);
}


template<class T, class CUDAVEC>
class base_vec2 : public CUDAVEC {
public:
	__host__ __device__
	base_vec2(){}

	__host__ __device__
	base_vec2(T c){
		x=c;
		y=c;
	}

	__host__ __device__
	base_vec2(T xx, T yy){
		x=xx;
		y=yy;
	}

	__host__ __device__
	base_vec2(const base_vec2<float, float2> &v){
		x=T(v.x);
		y=T(v.y);
	}

	__host__ __device__
	base_vec2(const CUDAVEC &v){
		x=v.x;
		y=v.y;
	}


};
template<class T, class CUDAVEC>
class base_vec3 : public CUDAVEC {
public:
	__host__ __device__
	base_vec3(){}

	__host__ __device__
	base_vec3(T c){
		x=c;
		y=c;
		z=c;
	}

	__host__ __device__
	base_vec3(T xx, T yy, T zz){
		x=xx;
		y=yy;
		z=zz;
	}

	__host__ __device__
	base_vec3(const CUDAVEC &v){
		x=v.x;
		y=v.y;
		z=v.z;
	}

	__host__ __device__
	base_vec3(const base_vec3<float, float3> &v){
		x=T(v.x);
		y=T(v.y);
		z=T(v.z);
	}
	__host__ __device__
	base_vec3(const base_vec3<int, int3> &v){
		x=T(v.x);
		y=T(v.y);
		z=T(v.z);
	}

	__host__ __device__
	base_vec3(const base_vec3<ushort, ushort3> &v){
		x=T(v.x);
		y=T(v.y);
		z=T(v.z);
	}
};
template<class T, class CUDAVEC>
class base_vec4 : public CUDAVEC {
public:
	//T &r; T &g; T &b; T &a;

	__host__ __device__
	base_vec4() /*: r(CUDAVEC::x), g(CUDAVEC::y), b(CUDAVEC::z), a(CUDAVEC::w)*/{}

	__host__ __device__
	base_vec4(T c) {
		x=c;
		y=c;
		z=c;
		w=c;
	}

	__host__ __device__
	base_vec4(T xx, T yy, T zz, T ww) {
		x=xx;
		y=yy;
		z=zz;
		w=ww;
	}

	__host__ __device__
	base_vec4(base_vec3<float, float3> v, T ww) {
		x=T(v.x);
		y=T(v.y);
		z=T(v.z);
		w=ww;
	}

	__host__ __device__
	base_vec4(const CUDAVEC &v)  {
		x=v.x;
		y=v.y;
		z=v.z;
		w=v.w;
	}

	__host__ __device__
	base_vec4(const base_vec4<float, float4> &v) {
		x=T(v.x);
		y=T(v.y);
		z=T(v.z);
		w=T(v.w);
	}

	/*__host__ __device__
	base_vec4& operator=(const base_vec4& d){
		CUDAVEC::operator=(d);

		return (*this);
	}*/

	/*__host__ __device__
	void operator=(base_vec4 d){
		x=d.x;
		y=d.y;
		z=d.z;
		w=d.w;
	}*/
};

typedef base_vec4<float, float4> vec4;
typedef base_vec3<float, float3> vec3;
typedef base_vec2<float, float2> vec2;

typedef base_vec4<int, int4> ivec4;
typedef base_vec3<int, int3> ivec3;
typedef base_vec2<int, int2> ivec2;

typedef base_vec4<uint, uint4> uvec4;
typedef base_vec3<uint, uint3> uvec3;
typedef base_vec2<uint, uint2> uvec2;

typedef base_vec4<unsigned short, ushort4> usvec4;
typedef base_vec3<unsigned short, ushort3> usvec3;
typedef base_vec2<unsigned short, ushort2> usvec2;

typedef base_vec4<short, short4> svec4;
typedef base_vec3<short, short3> svec3;
typedef base_vec2<short, short2> svec2;

inline __host__ __device__ float3 min(float3 a, float3 b)
{
    return fminf(a, b);
}
inline __host__ __device__ float2 min(float2 a, float2 b)
{
    return fminf(a, b);
}


inline __host__ __device__ float3 max(float3 a, float3 b)
{
    return fmaxf(a, b);
}

inline __host__ __device__ float2 max(float2 a, float2 b)
{
    return fmaxf(a, b);
}


template<class T, class CUDAVEC, class TO>
inline __host__ __device__ base_vec3<T, CUDAVEC> operator<<(base_vec3<T, CUDAVEC> v, TO o)
{
    return base_vec3<T, CUDAVEC>(v.x<<o, v.y<<o, v.z<<o);
}

template<class T, class CUDAVEC, class TO>
inline __host__ __device__ base_vec3<T, CUDAVEC> operator>>(base_vec3<T, CUDAVEC> v, TO o)
{
    return base_vec3<T, CUDAVEC>(v.x>>o, v.y>>o, v.z>>o);
}


template<class T, class CUDAVEC>
inline __host__ __device__ base_vec3<T, CUDAVEC> operator&(base_vec3<T, CUDAVEC> v1, base_vec3<T, CUDAVEC> v2)
{
    return base_vec3<T, CUDAVEC>(v1.x&v2.x, v1.y&v2.y, v1.z&v2.z);
}

template<class T, class CUDAVEC>
inline __host__ __device__ base_vec3<T, CUDAVEC> operator-(base_vec3<T, CUDAVEC> v1, base_vec3<T, CUDAVEC> v2)
{
    return base_vec3<T, CUDAVEC>(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);
}
template<class T, class CUDAVEC>
inline __host__ __device__ base_vec3<T, CUDAVEC> operator+(base_vec3<T, CUDAVEC> v1, base_vec3<T, CUDAVEC> v2)
{
    return base_vec3<T, CUDAVEC>(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);
}

/*inline __host__ __device__ int3 operator>>(int3 v1, int3 v2)
{
    return make_int3(v1.x>>v2.x, v1.y>>v2.y, v1.z>>v2.z);
}
inline __host__ __device__ int3 operator>>(int3 v1, int o)
{
    return make_int3(v1.x>>o, v1.y>>o, v1.z>>o);
}

inline __host__ __device__ int3 operator&(int3 v1, int3 v2)
{
    return make_int3(v1.x&v2.x, v1.y&v2.y, v1.z&v2.z);
}*/

//
inline __host__ __device__ int2 operator/(int2 v, int o)
{
    return make_int2(v.x/o, v.y/o);
}
inline __host__ __device__ int2 operator/(int2 v, int2 o)
{
    return make_int2(v.x/o.x, v.y/o.y);
}

//////////MATHS FUNC///////////

inline __host__ __device__ float radians(float deg){
	return deg/180.0f*CUGLSL_PI;
}

inline __host__ __device__ float3 floor(float3 v){
	return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

inline __host__ __device__ float3 step(float edge, float3 x){
	return make_float3(	x.x < edge ? 0.0f : 1.0f,
						x.y < edge ? 0.0f : 1.0f,
						x.z < edge ? 0.0f : 1.0f );
}

inline __host__ __device__ float3 fract(float3 v){
	return v-floor(v);
	//x - truncf(x) ?
}

//New
inline __device__ uint countLZ(uint x){
	return __clz(x);
}
inline __device__ uint bitfieldReverse(uint x){
	return __brev(x);
}

inline __device__ int findMSB(uint x){
	return 31-int(__clz(x));
}

inline __device__ int findLSB(uint x){
	return __clz( __brev(x) ); //should return -1 when returning 32...

}

template<class T>
__device__ int bitCount(T val){
	return __popc(val);
}

inline __device__ unsigned short floatToHalf(float x){
	return __float2half_rn(x);
}

inline __device__ float halfToFloat(unsigned short x){
	return __half2float(x);
}

//////////MATRIX TYPE///////////

typedef struct {
	float4 m[3];
} float3x4;


// transform vector by matrix (no translation)
__device__
inline float3 mulRot(const float3x4 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
__device__ __host__
inline float4 mul(const float3x4 &M, const float4 &v) {
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

typedef struct {
	union {
		float4 m[4];
		float _array[16];
	};
	float &element (int row, int col) {
		return _array[row | (col<<2)];
	}
	float element (int row, int col) const {
		return _array[row | (col<<2)];
	}
} float4x4;

__device__ __host__
inline float4 mul(const float4x4 &M, const float4 &v) {
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = dot(v, M.m[3]);

	return r;
}
__device__ __host__
inline float4 operator*(const float4x4 &M, const float4 &v) {
	return mul(M, v);
}

__device__ __host__
inline float3 mul(const float4x4 &M, const float3 &v0) {
	float4 r=mul(M, make_float4(v0.x, v0.y, v0.z, 1.0f));
	return make_float3(r.x/r.w, r.y/r.w, r.z/r.w);
}

//__device__ __host__
//inline float3 mulRot(const float4x4 &M, const float3 &v) {
//	float3 r;
//	r.x = dot(v, make_float3(M.m[0]));
//	r.y = dot(v, make_float3(M.m[1]));
//	r.z = dot(v, make_float3(M.m[2]));
//
//	return r;
//}

__device__ __host__
inline float3 mulRot(const float4x4 &M, const float3 &v) {
	float3 r;
	r.x = dot(v, make_float3(M._array[ 0], M._array[ 4], M._array[ 8]));
	r.y = dot(v, make_float3(M._array[ 1], M._array[ 5], M._array[ 9]));
	r.z = dot(v, make_float3(M._array[ 2], M._array[ 6], M._array[10]));
	return r;
}

__device__ __host__
inline float det(const float4x4 &mat)  {
	float det;
	det = mat._array[0] * mat._array[5] * mat._array[10];
	det += mat._array[4] * mat._array[9] * mat._array[2];
	det += mat._array[8] * mat._array[1] * mat._array[6];
	det -= mat._array[8] * mat._array[5] * mat._array[2];
	det -= mat._array[4] * mat._array[1] * mat._array[10];
	det -= mat._array[0] * mat._array[9] * mat._array[6];

	return det;
}

__host__
inline float4x4 transpose(const float4x4 &mat) {
	float4x4 ret;

	ret._array[0] = mat._array[0]; ret._array[1] = mat._array[4]; ret._array[2] = mat._array[8]; ret._array[3] = mat._array[12];
	ret._array[4] = mat._array[1]; ret._array[5] = mat._array[5]; ret._array[6] = mat._array[9]; ret._array[7] = mat._array[13];
	ret._array[8] = mat._array[2]; ret._array[9] = mat._array[6]; ret._array[10] = mat._array[10]; ret._array[11] = mat._array[14];
	ret._array[12] = mat._array[3]; ret._array[13] = mat._array[7]; ret._array[14] = mat._array[11]; ret._array[15] = mat._array[15];

	return ret;
}
__host__
inline float4x4 inverse(const float4x4 &mat) {
#if 0
	float4x4 ret;

	float idet = 1.0f / det(mat);
	ret._array[0] =  (mat._array[5] * mat._array[10] - mat._array[9] * mat._array[6]) * idet;
	ret._array[1] = -(mat._array[1] * mat._array[10] - mat._array[9] * mat._array[2]) * idet;
	ret._array[2] =  (mat._array[1] * mat._array[6] - mat._array[5] * mat._array[2]) * idet;
	ret._array[3] = 0.0;
	ret._array[4] = -(mat._array[4] * mat._array[10] - mat._array[8] * mat._array[6]) * idet;
	ret._array[5] =  (mat._array[0] * mat._array[10] - mat._array[8] * mat._array[2]) * idet;
	ret._array[6] = -(mat._array[0] * mat._array[6] - mat._array[4] * mat._array[2]) * idet;
	ret._array[7] = 0.0;
	ret._array[8] =  (mat._array[4] * mat._array[9] - mat._array[8] * mat._array[5]) * idet;
	ret._array[9] = -(mat._array[0] * mat._array[9] - mat._array[8] * mat._array[1]) * idet;
	ret._array[10] =  (mat._array[0] * mat._array[5] - mat._array[4] * mat._array[1]) * idet;
	ret._array[11] = 0.0;
	ret._array[12] = -(mat._array[12] * ret._array[0] + mat._array[13] * ret._array[4] + mat._array[14] * ret._array[8]);
	ret._array[13] = -(mat._array[12] * ret._array[1] + mat._array[13] * ret._array[5] + mat._array[14] * ret._array[9]);
	ret._array[14] = -(mat._array[12] * ret._array[2] + mat._array[13] * ret._array[6] + mat._array[14] * ret._array[10]);
	ret._array[15] = 1.0;

	return ret;
#else
	float4x4 minv;
	minv._array[0]=minv._array[1]=minv._array[2]=minv._array[3]=
		minv._array[4]=minv._array[5]=minv._array[6]=minv._array[7]=
		minv._array[8]=minv._array[9]=minv._array[10]=minv._array[11]=
		minv._array[12]=minv._array[13]=minv._array[14]=minv._array[15]=0;

		float r1[8], r2[8], r3[8], r4[8];
		float *s[4], *tmprow;

		s[0] = &r1[0];
		s[1] = &r2[0];
		s[2] = &r3[0];
		s[3] = &r4[0];

		register int i,j,p,jj;
		for(i=0;i<4;i++) {
			for(j=0;j<4;j++) {
				s[i][j] = mat.element(i,j);
				if(i==j) s[i][j+4] = 1.0;
				else	 s[i][j+4] = 0.0;
			}
		}
		float scp[4];
		for(i=0;i<4;i++) {
			scp[i] = float(fabs(s[i][0]));
			for(j=1;j<4;j++)
				if(float(fabs(s[i][j])) > scp[i]) scp[i] = float(fabs(s[i][j]));
			if(scp[i] == 0.0) return minv; // singular matrix!
		}

		int pivot_to;
		float scp_max;
		for(i=0;i<4;i++) {
			// select pivot row
			pivot_to = i;
			scp_max = float(fabs(s[i][i]/scp[i]));
			// find out which row should be on top
			for(p=i+1;p<4;p++)
				if (float(fabs(s[p][i]/scp[p])) > scp_max) {
					scp_max = float(fabs(s[p][i]/scp[p]));
					pivot_to = p;
				}
			// Pivot if necessary
			if(pivot_to != i) {
				tmprow = s[i];
				s[i] = s[pivot_to];
				s[pivot_to] = tmprow;
				float tmpscp;
				tmpscp = scp[i];
				scp[i] = scp[pivot_to];
				scp[pivot_to] = tmpscp;
			}

			float mji;
			// perform gaussian elimination
			for(j=i+1;j<4;j++) {
				mji = s[j][i]/s[i][i];
				s[j][i] = 0.0;
				for(jj=i+1;jj<8;jj++)
					s[j][jj] -= mji*s[i][jj];
			}
		}
		if(s[3][3] == 0.0) return minv; // singular matrix!

		float mij;
		for(i=3;i>0;i--) {
			for(j=i-1;j > -1; j--) {
				mij = s[j][i]/s[i][i];
				for(jj=j+1;jj<8;jj++)
					s[j][jj] -= mij*s[i][jj];
			}
		}

		for(i=0;i<4;i++)
			for(j=0;j<4;j++)
				minv.element(i,j) = s[i][j+4] / s[i][i];


		return minv;
#endif
}


typedef float4x4 mat4;


/////////////HIGH LEVEL FUNCS/////////////
template<class T>
__device__ T warpReduceMin(T val) {
    // Seed starting value as inverse lane ID
    T value = val; //31 - laneId;

    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2)
        //value += __shfl_xor(value, i, 32);
		value = min(value, __shfl_xor(value, i, 32));

  return value;
}

template<class T>
__device__ T warpReduceMax(T val) {
    // Seed starting value as inverse lane ID
    T value = val; //31 - laneId;

    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2)
        //value += __shfl_xor(value, i, 32);
		value = max(value, __shfl_xor(value, i, 32));

  return value;
}

template<class T>	//Width version
__device__ T warpReduceMax(T val, int width) {
    // Seed starting value as inverse lane ID
    T value = val; //31 - laneId;

    // Use XOR mode to perform butterfly reduction
    for (int i=(width/2); i>=1; i/=2)
        //value += __shfl_xor(value, i, 32);
		value = max(value, __shfl_xor(value, i, width));

  return value;
}

template<class T>
__device__ T warpReduceAND(T val) {
    // Seed starting value as inverse lane ID
    T value = val; //31 - laneId;

    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2)
		value = value & __shfl_xor(value, i, 32);

  return value;
}

template<class T>
__device__ T warpReduceXOR(T val) {
    // Seed starting value as inverse lane ID
    T value = val; //31 - laneId;

    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2)
		value = value ^ __shfl_xor(value, i, 32);

  return value;
}
#endif