#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <cassert>
#include <math.h>
#include <vector>
#include <array>
#include <string>
#include <random>
#include <thread>
#include <chrono>
#include <stdio.h>
#include <assert.h>

#define DATA_NUM		(60000)
#define DATA_SIZE_X		(32)
#define DATA_SIZE_Y		(32)
#define DATA_SIZE_Z		(3)
#define DATA_SIZE		(DATA_SIZE_X * DATA_SIZE_Y * DATA_SIZE_Z)

#define WEIGHT_UPDATE_FUN(x)		(x / (1.f + (x * x)))
//#define WEIGHT_UPDATE_FUN(x)		(x)

using namespace std;
unsigned char labels[DATA_NUM];
unsigned char images[DATA_NUM][DATA_SIZE];
string names[10] = {
	"Airplane",
	"Automobile",
	"Bird",
	"Cat",
	"Deer",
	"Dog",
	"Frog",
	"Horse",
	"Ship",
	"Truck"
};

//int N = 1;
const int   Bach = 16;
const int   Epoch = 10;
float U = 0.001f;
float H = 0.00f;

random_device seed_gen;
mt19937_64 engine(seed_gen());
normal_distribution<float> randN(0.f, 1.f);
uniform_real_distribution<float> randU(0.f, 1.f);

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}


float FloatAve(float x)
{
	static int cur = 0;
	static float sum = 0.f;
	static array<float, 20> tem;
	sum += x;
	sum -= tem[cur];
	tem[cur] = x;
	cur = (cur + 1) % tem.size();
	return sum / tem.size();
}

// すべてにおいてブロックのz座標はデータの番号に、xy方向のブロック*スレッドのサイズは更新されるデータと一致
// プーリングのみ出力データに合わせる

// 順伝播に関して
// 各全結合は、前の層の各xy平面を取り出し、l個のノードへ全結合する
// スレッドのx座標はそのl個のノードの位置を表し、ブロックのx座標は使うxy平面の位置を表す
// 全結合系&ソフトマックスは、x座標のみで出力の位置を表す
// ソフトマックスはスレッドのz方向でも並列可能

// 逆伝播に関して
// 逆伝播の関数は、順伝播のために確保したメモリを再利用するため、
// 誤差関数の入力変数微分は一時メモリに保存したのち一斉に更新する
// 重みの更新に入力データと誤差の出力データ微分が必要なので、先に重みを計算する
// 重みの更新は重みごとにすべてのデータに対して計算する
// 重みの更新をするカーネルは基本Block(ix,iy,iz)Thread(ox,oy,oz)
// 畳み込みはBlock(fx,fy,fz)Thread(oz,di,1)

/*
* flz := フィルタのz方向の大きさ
* fr* := フィルタの*方向の半径 ( 3 * 3　なら 1 )
* il* := 入力データの*方向の大きさ
* ol* := 出力データの*方向の大きさ
* d := 次数
* n := 計算するデータ数
*/
__global__ void Convolution(const float* In, float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olz, int oly, int olx, int flz, int fry, int frx)
{
	const int thz =                          threadIdx.z;
	const int oy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ox = blockDim.x * blockIdx.x + threadIdx.x;
	const int flx = (2 * frx + 1);
	const int fly = (2 * fry + 1);
	int Iindex;
	int Oindex;
	int Nindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		for (int oz = thz; oz < olz; oz += blockDim.z) {
			Oindex = ((dn * olz + oz) * oly + oy) * olx + ox;
			for (int iz = 0; iz < ilz; iz++)
			for (int iy = max(0, oy - fry); iy < min(ily, oy + fry + 1); iy++)
			for (int ix = max(0, ox - frx); ix < min(ilx, ox + frx + 1); ix++) {
				Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
				Nindex = ((oz * flz + iz) * fly + (iy - oy + fry)) * flx + (ix - ox + frx);
				Out[Oindex] += NN[Nindex] * In[Iindex];
			}
		}
	}
	return;
}
__global__ void RevConvolutionN(float* dNN, const float* In, const float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olz, int oly, int olx, int flz, int fry, int frx)
{
	const int fz = blockIdx.z;
	const int fy = blockIdx.y;
	const int fx = blockIdx.x;
	const int oz = threadIdx.x;
	const int flx = (2 * frx + 1);
	const int fly = (2 * fry + 1);
	int Iindex;
	int Oindex;
	int Nindex;
	float sum = 0.f;
	Nindex = ((oz * flz + fz) * fly + fy) * flx + fx;
	for (int dn = 0; dn < n; dn++) {
		for (int iy = max(0, fry - fy); iy < ily + min(0, fry - fy); iy++)
		for (int ix = max(0, frx - fx); ix < ilx + min(0, frx - fx); ix++) {
			Iindex = ((dn * ilz + fx) * ily + iy) * ilx + ix;
			Oindex = ((dn * olz + oz) * oly + (iy + fy - fry)) * olx + (ix + fx - frx);
			sum += In[Iindex] * Out[Oindex];
		}
	}
	dNN[Nindex] = sum / n;
}
__global__ void RevConvolutionL(float* Tem, const float* In, const float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olz, int oly, int olx, int flz, int fry, int frx)
{
	const int thz =                          threadIdx.z;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int flx = (2 * frx + 1);
	const int fly = (2 * fry + 1);
	int Iindex;
	int Oindex;
	int Nindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		for (int iz = thz; iz < ilz; iz += blockDim.z) {
			Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
			Tem[Iindex] = 0.f;
			for (int oz = 0; oz < olz; oz++)
			for (int oy = max(0, iy - fry); oy < min(oly, iy + fry + 1); oy++)
			for (int ox = max(0, ix - frx); ox < min(olx, ix + frx + 1); ox++) {
				Oindex = ((dn * olz + oz) * oly + oy) * olx + ox;
				Nindex = ((oz * flz + iz) * fly + (iy - oy + fry)) * flx + (ix - ox + frx);
				Tem[Iindex] += NN[Nindex] * Out[Oindex];
			}
		}
	}
	return;
}
__global__ void ConvolutionT(const float* In, float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olz, int oly, int olx, int flz, int fry, int frx, int d)
{
	const int thz =                          threadIdx.z;
	const int oy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ox = blockDim.x * blockIdx.x + threadIdx.x;
	const int flx = (2 * frx + 1);
	const int fly = (2 * fry + 1);
	int Iindex;
	int Oindex;
	int Nindex;
	float p, t;
	/*const int slx = blockDim.y + flx - 1;
	const int sly = blockDim.x + fly - 1;
	const int irxl = max(0 , int(blockDim.x *  blockIdx.x)       - frx);
	const int irxh = min(lx, int(blockDim.x * (blockIdx.x + 1u)) + frx);
	const int iryl = max(0 , int(blockDim.y *  blockIdx.y)       - fry);
	const int iryh = min(ly, int(blockDim.y * (blockIdx.y + 1u)) + fry);
	__shared__ float InBlock[1000];
	for (int iz = z; iz < flz; iz += blockDim.z) {
		for (int dx = -1; dx <= 1; dx++)
		for (int dy = -1; dy <= 1; dy++) {
			int thx = dx * blockDim.x + threadIdx.x;
			int thy = dy * blockDim.y + threadIdx.y;
			int px = thx + blockDim.x * blockIdx.x;
			int py = thy + blockDim.y * blockIdx.y;
			if (irxl <= px && px < irxh && iryl <= py && py < iryh) {
				InBlock[(iz * sly + thy + fry) * slx + thx + frx] = In[(iz * ly + py) * lx + px];
			}
		}
	}
	__syncthreads();//*/
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		for (int oz = thz; oz < olz; oz += blockDim.z) {
			Oindex = ((dn * olz + oz) * oly + oy) * olx + ox;
			for (int iz = 0; iz < ilz; iz++)
			for (int iy = max(0, oy - fry); iy < min(ily, oy + fry + 1); iy++)
			for (int ix = max(0, ox - frx); ix < min(ilx, ox + frx + 1); ix++) {
				Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
				Nindex = (((oz * flz + iz) * fly + (iy - oy + fry)) * flx + (ix - ox + frx)) * d;
				t = 0.f;
				p = In[Iindex];
				for (int di = 0; di < d; di++) {
					Out[Oindex] += NN[Nindex + di] * cosf(t);
					t += p;
				}
			}
		}
	}
	return;
}
__global__ void RevConvolutionTN(float* dNN, const float* In, const float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olz, int oly, int olx, int flz, int fry, int frx, int d)
{
	const int fz = blockIdx.z;
	const int fy = blockIdx.y;
	const int fx = blockIdx.x;
	const int oz = threadIdx.x;
	const int di = threadIdx.y;
	const int flx = (2 * frx + 1);
	const int fly = (2 * fry + 1);
	int Iindex;
	int Oindex;
	int Nindex;
	float sum = 0.f;
	Nindex = (((oz * flz + fz) * fly + fy) * flx + fx) * d + di;
	for (int dn = 0; dn < n; dn++) {
		for (int iy = max(0, fry - fy); iy < ily + min(0, fry - fy); iy++)
		for (int ix = max(0, frx - fx); ix < ilx + min(0, frx - fx); ix++) {
			Iindex = ((dn * ilz + fx) * ily + iy) * ilx + ix;
			Oindex = ((dn * olz + oz) * oly + (iy + fy - fry)) * olx + (ix + fx - frx);
			sum += cosf(di * In[Iindex]) * Out[Oindex];
		}
	}
	dNN[Nindex] = sum / n;
	/*__shared__ float tem[1000];
	for (int dn = 0; dn < n; dn++) {
		for (int iy = max(0, fry - fy); iy < ily + min(0, fry - fy); iy++)
		for (int ix = max(0, frx - fx); ix < ilx + min(0, frx - fx); ix++) {
			Iindex = ((dn * ilz + fx) * ily + iy) * ilx + ix;
			t = 0.f;
			p = In[Iindex] / d;
			for (int di = 0; di < d; di++) {
				for (int oz = thx; oz < olz; oz += blockDim.x) {
					Oindex = ((dn * olz + oz) * oly + (iy + fy - fry)) * olx + (ix + fx - frx);
					//Nindex = (((oz * flz + fz) * fly + fy) * flx + fx) * d;
					tem[oz * d + di] += cosf(t) * Out[Oindex];
				}
				t += p;
			}
		}
	}//*/
}
__global__ void RevConvolutionTL(float* Tem, const float* In, const float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olz, int oly, int olx, int flz, int fry, int frx, int d)
{
	const int thz =                          threadIdx.z;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int flx = (2 * frx + 1);
	const int fly = (2 * fry + 1);
	int Iindex;
	int Oindex;
	int Nindex;
	float p, t, dsum;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		for (int iz = thz; iz < ilz; iz += blockDim.z) {
			Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
			Tem[Iindex] = 0.f;
			for (int oz = 0; oz < olz; oz++)
			for (int oy = max(0, iy - fry); oy < min(oly, iy + fry + 1); oy++)
			for (int ox = max(0, ix - frx); ox < min(olx, ix + frx + 1); ox++) {
				Oindex = ((dn * olz + oz) * oly + oy) * olx + ox;
				Nindex = (((oz * flz + iz) * fly + (iy - oy + fry)) * flx + (ix - ox + frx)) * d;
				dsum = 0.f;
				t = 0.f;
				p = In[Iindex];
				for (int di = 0; di < d; di++) {
					dsum += di * NN[Nindex + di] * sinf(t);
					t += p;
				}
				Tem[Iindex] += dsum * Out[Oindex];
			}
			Tem[Iindex] = -Tem[Iindex];
		}
	}
	return;
}
/*
* f* := プーリングの*方向の大きさ
* il* := 入力データの*方向の大きさ
* ol* := 出力データの*方向の大きさ
* n := 計算するデータ数
*/
__global__ void APooling(const float* In, float* Out, int n, int ilz, int ily, int ilx, int olz, int oly, int olx, int fy, int fx)
{
	const int thz = threadIdx.z;
	const int oy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ox = blockDim.x * blockIdx.x + threadIdx.x;
	int Iindex;
	int Oindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		for (int ioz = thz; ioz < olz; ioz += blockDim.z) {
			Oindex = ((dn * olz + ioz) * oly + oy) * olx + ox;
			for (int dy = 0; dy < fy; dy++)
			for (int dx = 0; dx < fx; dx++) {
				Iindex = ((dn * ilz + ioz) * ily + oy * fy + dy) * ilx + ox * fx + dx;
				Out[Oindex] += In[Iindex];
			}
			Out[Oindex] /= fy * fx;
		}
	}
	return;
}
__global__ void RevAPoolingL(float* Tem, const float* In, const float* Out, int n, int ilz, int ily, int ilx, int olz, int oly, int olx, int fy, int fx)
{
	const int thz = threadIdx.z;
	const int oy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ox = blockDim.x * blockIdx.x + threadIdx.x;
	int Iindex;
	int Oindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		for (int ioz = thz; ioz < olz; ioz += blockDim.z) {
			Oindex = ((dn * olz + ioz) * oly + oy) * olx + ox;
			for (int dy = 0; dy < fy; dy++)
			for (int dx = 0; dx < fx; dx++) {
				Iindex = ((dn * ilz + ioz) * ily + oy * fy + dy) * ilx + ox * fx + dx;
				Tem[Iindex] = Out[Oindex] / (fy * fx);
			}
		}
	}
	return;
}
__global__ void MPooling(const float* In, float* Out, int n, int ilz, int ily, int ilx, int olz, int oly, int olx, int fy, int fx)
{
	const int thz =                          threadIdx.z;
	const int oy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ox = blockDim.x * blockIdx.x + threadIdx.x;
	int Oindex;
	for (int i = blockIdx.z; i < n; i += gridDim.z) {
		for (int oz = thz; oz < olz; oz += blockDim.z) {
			Oindex = ((i * olz + oz) * oly + oy) * olx + ox;
			Out[Oindex] = -FLT_MAX;
			for (int dy = 0; dy < fy; dy++)
			for (int dx = 0; dx < fx; dx++) {
				Out[Oindex] = max(Out[Oindex], In[((i * ilz + oz) * ily + oy * fy + dy) * ilx + ox * fx + dx]);
			}
		}
	}
	return;
}
/*
* il* := 入力データの*方向の大きさ
* olb := 出力データの1層当たりの大きさ
* d := 次数
* n := 計算するデータ数
*/
__global__ void EachDenseT(const float* In, float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olb, int d)
{
	const int iz = blockIdx.x;
	const int ox = threadIdx.x;
	int Iindex;
	int Oindex;
	int Nindex;
	float p, t;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		Oindex = (dn * ilz + iz) * olb + ox;
		for (int iy = 0; iy < ily; iy++)
		for (int ix = 0; ix < ilx; ix++) {
			Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
			Nindex = (((iz * ily + iy) * ilx + ix) * olb + ox) * d;
			t = 0.f;
			p = In[Iindex] / d;
			for (int di = 0; di < d; di++) {
				Out[Oindex] += NN[Nindex + di] * cosf(t);
				t += p;
			}
		}
	}
	return;
}
__global__ void RevEachDenseTN(float* dNN, const float* In, float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olb, int d)
{
	const int iz = blockIdx.z;
	const int iy = blockIdx.y;
	const int ix = blockIdx.x;
	const int ox = threadIdx.x;
	const int di = threadIdx.y;
	int Iindex;
	int Oindex;
	int Nindex;
	float sum = 0.f;
	Nindex = (((iz * ily + iy) * ilx + ix) * olb + ox) * d + di;
	for (int dn = 0; dn < n; dn++) {
		Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
		Oindex = (dn * ilz + iz) * olb + ox;
		sum += cosf(di * In[Iindex] / d) * Out[Oindex];
	}
	dNN[Nindex] = sum / n;
	return;
}
__global__ void RevEachDenseTL(float* Tem, const float* In, float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olb, int d)
{
	const int thz = threadIdx.z;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int Iindex;
	int Oindex;
	int Nindex;
	float p, t, dsum;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		for (int iz = thz; iz < ilz; iz += blockDim.z) {
			Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
			Tem[Iindex] = 0.f;
			for (int ox = 0; ox < olb; ox++) {
				Oindex = (dn * ilz + iz) * olb + ox;
				Nindex = (((iz * ily + iy) * ilx + ix) * olb + ox) * d;
				dsum = 0.f;
				t = 0.f;
				p = In[Iindex] / d;
				for (int di = 0; di < d; di++) {
					dsum += di * NN[Nindex + di] * sinf(t);
					t += p;
				}
				Tem[Iindex] += dsum * Out[Oindex];
			}
			Tem[Iindex] /= -d;
		}
	}
	return;
}
/*
* il* := 入力データの*方向の大きさ
* olx := 出力データのx方向の大きさ
* d := 次元
* n := 計算するデータ数
*/
__global__ void Dense(const float* In, float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olx)
{
	const int ox = blockDim.x * blockIdx.x + threadIdx.x;
	int Iindex;
	int Oindex;
	int Nindex;
	float t;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		Oindex = dn * olx + ox;
		t = 0.f;
		for (int iz = 0; iz < ilz; iz++)
		for (int iy = 0; iy < ily; iy++)
		for (int ix = 0; ix < ilx; ix++) {
			Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
			Nindex = ((iz * ily + iy) * ilx + ix) * olx + ox;
			t += NN[Nindex] * In[Iindex];
		}
		Out[Oindex] += t;
	}
	return;
}
__global__ void RevDenseN(float* dNN, const float* In, const float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olx)
{
	const int iz = blockIdx.z;
	const int iy = blockIdx.y;
	const int ix = blockIdx.x;
	const int ox = threadIdx.x;
	int Iindex;
	int Oindex;
	int Nindex;
	float sum = 0.f;
	Nindex = ((iz * ily + iy) * ilx + ix) * olx + ox;
	for (int dn = 0; dn < n; dn++) {
		Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
		Oindex = dn * olx + ox;
		sum += In[Iindex] * Out[Oindex];
	}
	dNN[Nindex] = sum / n;
	return;
}
__global__ void RevDenseL(float* Tem, const float* In, const float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olx)
{
	const int thz = threadIdx.z;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int Iindex;
	int Oindex;
	int Nindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		for (int iz = thz; iz < ilz; iz += blockDim.z) {
			Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
			Tem[Iindex] = 0.f;
			for (int ox = 0; ox < olx; ox++) {
				Oindex = dn * olx + ox;
				Nindex = ((iz * ily + iy) * ilx + ix) * olx + ox;
				Tem[Iindex] += NN[Nindex] * Out[Oindex];
			}
		}
	}
	return;
}
__global__ void DenseT(const float* In, float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olx, int d)
{
	const int ox = blockDim.x * blockIdx.x + threadIdx.x;
	int Iindex;
	int Oindex;
	int Nindex;
	float p, t;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		Oindex = dn * olx + ox;
		for (int iz = 0; iz < ilz; iz++)
		for (int iy = 0; iy < ily; iy++)
		for (int ix = 0; ix < ilx; ix++) {
			Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
			Nindex = (((iz * ily + iy) * ilx + ix) * olx + ox) * d;
			t = 0.f;
			p = In[Iindex];
			for (int di = 0; di < d; di++) {
				Out[Oindex] += NN[Nindex + di] * cosf(t);
				t += p;
			}
		}
	}
	return;
}
__global__ void RevDenseTN(float* dNN, const float* In, const float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olx, int d)
{
	const int iz = blockIdx.z;
	const int iy = blockIdx.y;
	const int ix = blockIdx.x;
	const int ox = threadIdx.x;
	const int di = threadIdx.y;
	int Iindex;
	int Oindex;
	int Nindex;
	float sum = 0.f;
	Nindex = (((iz * ily + iy) * ilx + ix) * olx + ox) * d + di;
	for (int dn = 0; dn < n; dn++) {
		Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
		Oindex = dn * olx + ox;
		sum += cosf(di * In[Iindex]) * Out[Oindex];
	}
	dNN[Nindex] = sum / n;
	return;
}
__global__ void RevDenseTL(float* Tem, const float* In, const float* Out, const float* NN, int n, int ilz, int ily, int ilx, int olx, int d)
{
	const int thz = threadIdx.z;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int Iindex;
	int Oindex;
	int Nindex;
	float p, t, dsum;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		for (int iz = thz; iz < ilz; iz += blockDim.z) {
			Iindex = ((dn * ilz + iz) * ily + iy) * ilx + ix;
			Tem[Iindex] = 0.f;
			for (int ox = 0; ox < olx; ox++) {
				Oindex = dn * olx + ox;
				Nindex = (((iz * ily + iy) * ilx + ix) * olx + ox) * d;
				t = 0.f;
				p = In[Iindex];
				dsum = 0.f;
				for (int di = 0; di < d; di++) {
					dsum += di * NN[Nindex + di] * sinf(t);
					t += p;
				}
				Tem[Iindex] += dsum * Out[Oindex];
			}
			Tem[Iindex] = -Tem[Iindex];
		}
	}//*/
	return;
}
/*
* n := 計算するデータ数
*/
__global__ void Softmax(const float* In, float* Out, int n, int ilx, int olx)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	int Iindex;
	int Oindex;
	float m, t;
	for (int i = z; i < n; i += gridDim.z * blockDim.z) {
		m = -FLT_MAX;
		t = 0.f;
		Oindex = i * olx + x;
		Iindex = i * ilx;
		for (int ix = 0; ix < ilx; ix++) {
			m = max(m, In[Iindex + ix]);
		}
		for (int ix = 0; ix < ilx; ix++) {
			t += expf(In[Iindex + ix] - m);
		}
		Out[Oindex] = expf(In[Iindex + x] - m) / t;
	}
	return;
}
__global__ void RevSoftmaxL(float* Tem, const float* In, const float* Out, int n, int ilx, int olx)
{
	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	int Iindex;
	int Oindex;
	for (int i = z; i < n; i += gridDim.z * blockDim.z) {
		Oindex = i * olx + x;
		Iindex = i * ilx + x;
		Tem[Iindex] = Out[Oindex];
	}
	return;
}
/*
* l* := 入出力データの*方向の大きさ
*/
__global__ void Bias(const float* In, float* Out, const float* NN, int n, int lz, int ly, int lx)
{
	const int z =                           threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	int IOindex;
	int Nindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		IOindex = ((dn * lz + z) * ly + y) * lx + x;
		Nindex = (z * ly + y) * lx + x;
		Out[IOindex] = In[IOindex] + NN[Nindex];
	}
	return;
}
__global__ void RevBiasN(float* dNN, const float* In, const float* Out, const float* NN, int n, int lz, int ly, int lx)
{
	const int z = blockIdx.z;
	const int y = blockIdx.y;
	const int x = blockIdx.x;
	int IOindex;
	int Nindex;
	Nindex = (z * ly + y) * lx + x;
	float sum = 0.f;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		IOindex = ((dn * lz + z) * ly + y) * lx + x;
		sum += Out[IOindex];
	}
	dNN[Nindex] = sum / n;
	return;
}
__global__ void RevBiasL(float* Tem, const float* In, const float* Out, const float* NN, int n, int lz, int ly, int lx)
{
	const int z =                           threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	int IOindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		IOindex = ((dn * lz + z) * ly + y) * lx + x;
		Tem[IOindex] = Out[IOindex];
	}
	return;
}
/*
* l* := 入出力データの*方向の大きさ
*/
__global__ void Relu(const float* In, float* Out, int n, int lz, int ly, int lx)
{
	const int z =                           threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	int IOindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		IOindex = ((dn * lz + z) * ly + y) * lx + x;
		Out[IOindex] = max(0.f, In[IOindex]);
	}
	return;
}
__global__ void RevReluL(float* Tem, const float* In, float* Out, int n, int lz, int ly, int lx)
{
	const int z =                           threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	int IOindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		IOindex = ((dn * lz + z) * ly + y) * lx + x;
		if (In[IOindex] > 0.f) Tem[IOindex] = Out[IOindex];
		else                   Tem[IOindex] = 0.f;
	}
	return;
}
/*
* l* := 入出力データの*方向の大きさ
*/
__global__ void Normalize(const float* In, float* Out, int n, int lz, int ly, int lx, float norm, float* Maximum)
{
	const int Index = threadIdx.x;
	const int stride = lz * ly * lx;
	__shared__ float tem[1024];
	tem[Index] = 0.f;
	int size, n1 = Index, n2 = Index;
	if (2 * Index < blockDim.x) n1 = 2 * Index;
	if (2 * Index + 1 < blockDim.x) n2 = 2 * Index + 1;
	for (int dn = blockIdx.x; dn < n; dn += gridDim.x) {
		__syncthreads();
		tem[Index] = 0.f;
		for (int i = Index; i < stride; i += blockDim.x) {
			tem[Index] = max(tem[Index], In[dn * stride + i]);
		}
		__syncthreads();
		size = blockDim.x;
		while (size != 1) {
			tem[Index] = max(tem[Index], tem[n1]);
			tem[Index] = max(tem[Index], tem[n2]);
			size = (size + 1) >> 1;
			__syncthreads();
		}
		if (Index == 0) Maximum[dn] = tem[0];
		for (int i = Index; i < stride; i += blockDim.x) {
			Out[dn * stride + i] = norm * In[dn * stride + i] / tem[0];
		}
	}
	return;
}
__global__ void RevNormalizeL(float* Tem, const float* In, const float* Out, int n, int lz, int ly, int lx, float norm, float* Maximum)
{
	const int Index = threadIdx.x;
	const int stride = lz * ly * lx;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		for (int i = Index; i < stride; i += blockDim.x) {
			Tem[dn * stride + i] = norm * Out[dn * stride + i] / Maximum[dn];
		}
	}
	return;
}
/*
* l := 出力データサイズ
* n := 計算するデータ数
*/
__global__ void CheckAns(float* Out, const float* Ans, int n, int l)
{
	int Index = blockDim.x * blockIdx.x + threadIdx.x;
	for (int dn = Index; dn < n * l; dn += gridDim.x * blockDim.x) {
		Out[dn] -= Ans[dn];
	}
	return;
}
/*
* h := 学習率
* u := 減衰率
* n := 重みの個数
*/
__global__ void UpdateWeight(float* NN, float* dNN, int n, float u, float h)
{
	const int Index = blockDim.x * blockIdx.x + threadIdx.x;
	float t;
	for (int dn = Index; dn < n; dn += gridDim.x * blockDim.x) {
		t = dNN[dn];
		NN[dn] -= u * WEIGHT_UPDATE_FUN(t);
		dNN[dn] = 0.f;
	}
	return;
}


class Layer
{
public:
	Layer(void) = default;
	~Layer(void) = default;
	uint3 GetInSize(void) const { return InSize; }
	uint3 GetOutSize(void) const { return OutSize; }
	virtual int GetMemSize(void) { return OutSize.z * OutSize.y * OutSize.x; }
	virtual bool CheckLayer(uint3 in_size) { return false; }
	virtual int GetNNSize(void) const { return -1; }
	virtual void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) { }
	virtual void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) { }
	virtual void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) { }

//protected:
	uint3 InSize, OutSize;
protected:
	bool InputFlag = false;
};
class InputLayer : public Layer
{
public:
	InputLayer(uint3 size)
	{
		InSize = OutSize = size;
		InputFlag = true;
	}
	~InputLayer(void) = default;
	bool CheckLayer(uint3 in_size = uint3()) override { return true; }
	int GetNNSize(void) const override { return 0; }
	void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) override { }
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override { }
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override { }

private:

};
class BiasLayer : public Layer
{
public:
	BiasLayer(void) = default;
	~BiasLayer(void) = default;
	bool CheckLayer(uint3 in_size) override { OutSize = InSize = in_size; return true; }
	int GetNNSize(void) const override { return InSize.z * InSize.y * InSize.x; }
	void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) override
	{
		Bias<<<Block, Thread, 0, stream>>>(in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override
	{
		dim3 Block = {InSize.x, InSize.y, InSize.z};
		RevBiasN<<<Block, 1, 0, stream>>>(dnn, in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x);
	}
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevBiasL<<<Block, Thread, 0, stream>>>(tem, in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x);
	}

private:

};
class DenseLayer : public Layer
{
public:
	DenseLayer(unsigned int out_size)
	{
		OutSize = {out_size, 1, 1};
	}
	~DenseLayer(void) = default;
	bool CheckLayer(uint3 in_size) override { InSize = in_size; return true; }
	int GetNNSize(void) const override { return InSize.z * InSize.y * InSize.x * OutSize.x; }
	void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) override
	{
		Dense<<<Block, Thread, 0, stream>>>(in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x, OutSize.x);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override
	{
		dim3 Block = {InSize.x, InSize.y, InSize.z}, Thread = {OutSize.x, 1, 1};
		RevDenseN<<<Block, Thread, 0, stream>>>(dnn, in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x, OutSize.x);
	}
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevDenseL<<<Block, Thread, 0, stream>>>(tem, in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x, OutSize.x);
	}

private:

};
class DenseTLayer : public Layer
{
public:
	DenseTLayer(unsigned int out_size, unsigned int d) : d(d)
	{
		OutSize = {out_size, 1, 1};
	}
	~DenseTLayer(void) = default;
	bool CheckLayer(uint3 in_size) override { InSize = in_size; return true; }
	int GetNNSize(void) const override { return InSize.z * InSize.y * InSize.x * OutSize.x * d; }
	void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) override
	{
		DenseT<<<Block, Thread, 0, stream>>>(in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x, OutSize.x, d);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override
	{
		dim3 Block = {InSize.x, InSize.y, InSize.z}, Thread = {OutSize.x, d, 1};
		RevDenseTN<<<Block, Thread, 0, stream>>>(dnn, in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x, OutSize.x, d);
	}
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevDenseTL<<<Block, Thread, 0, stream>>>(tem, in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x, OutSize.x, d);
	}

private:
	unsigned int d;

};
class ReluLayer : public Layer
{
public:
	ReluLayer(void) = default;
	~ReluLayer(void) = default;
	bool CheckLayer(uint3 in_size) override { OutSize = InSize = in_size; return true; }
	int GetNNSize(void) const override { return 0; }
	void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) override
	{
		Relu<<<Block, Thread, 0, stream>>>(in_layer, out_layer, M, InSize.z, InSize.y, InSize.x);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override { }
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevReluL<<<Block, Thread, 0, stream>>>(tem, in_layer, out_layer, M, InSize.z, InSize.y, InSize.x);
	}

private:

};
class ConvolutionLayer : public Layer
{
public:
	ConvolutionLayer(unsigned int frx, unsigned int fry, unsigned int num) : frx(frx), fry(fry), num(num) { }
	~ConvolutionLayer(void) = default;
	bool CheckLayer(uint3 in_size) override
	{
		OutSize = InSize = in_size;
		OutSize.z = num;
		if (2 * fry + 1 <= in_size.y && 2 * frx + 1 <= in_size.x) return true;
		return false;
	}
	int GetNNSize(void) const override { return num * InSize.z * fry * frx; }
	void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) override
	{
		Convolution<<<Block, Thread, 0, stream>>>(in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x, OutSize.z, OutSize.y, OutSize.x, InSize.z, fry, frx);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override
	{
		dim3 Block = {frx, fry, InSize.z}, Thread = {num, 1, 1};
		RevConvolutionN<<<Block, Thread, 0, stream>>>(dnn, in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x, OutSize.z, OutSize.y, OutSize.x, InSize.z, fry, frx);
	}
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevConvolutionL<<<Block, Thread, 0, stream>>>(tem, in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x, OutSize.z, OutSize.y, OutSize.x, InSize.z, fry, frx);
	}

private:
	unsigned int fry, frx, num;

};
class APoolingLayer : public Layer
{
public:
	APoolingLayer(int fx, int fy) : fx(fx), fy(fy) { }
	~APoolingLayer(void) = default;
	bool CheckLayer(uint3 in_size) override
	{
		OutSize = InSize = in_size;
		OutSize.x /= fx;
		OutSize.y /= fy;
		if (in_size.y % fy == 0 && in_size.x % fx == 0) return true;
		return false;
	}
	int GetNNSize(void) const override { return 0; }
	void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) override
	{
		APooling <<<Block, Thread, 0, stream>>>(in_layer, out_layer, M, InSize.z, InSize.y, InSize.x, OutSize.z, OutSize.y, OutSize.x, fy, fx);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override { }
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevAPoolingL<<<Block, Thread, 0, stream>>>(tem, in_layer, out_layer, M, InSize.z, InSize.y, InSize.x, OutSize.z, OutSize.y, OutSize.x, fy, fx);
	}

private:
	int fx, fy;

};
class SoftmaxLayer : public Layer
{
public:
	SoftmaxLayer(void) = default;
	~SoftmaxLayer(void) = default;
	bool CheckLayer(uint3 in_size) override
	{
		OutSize = InSize = in_size;
		if (in_size.y == 1 && in_size.z == 1) return true;
		return false;
	}
	int GetNNSize(void) const override { return 0; }
	void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) override
	{
		Softmax<<<Block, Thread, 0, stream>>>(in_layer, out_layer, M, InSize.x, OutSize.x);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override { }
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevSoftmaxL<<<Block, Thread, 0, stream>>>(tem, in_layer, out_layer, M, InSize.x, OutSize.x);
	}

private:

};
class NormalizeLayer : public Layer
{
public:
	NormalizeLayer(float norm = 1.f) : norm(norm) { }
	~NormalizeLayer(void) = default;
	bool CheckLayer(uint3 in_size) override { OutSize = InSize = in_size; return true; }
	int GetNNSize(void) const override { return 0; }
	int GetMemSize(void) override { return OutSize.z * OutSize.y * OutSize.x + 1; }
	void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) override
	{
		Normalize<<<Block, Thread, 0, stream>>>(in_layer, &out_layer[N], M, InSize.z, InSize.y, InSize.x, norm, out_layer);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override { }
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevNormalizeL<<<Block, Thread, 0, stream>>>(tem, in_layer, &out_layer[N], M, InSize.z, InSize.y, InSize.x, norm, out_layer);
	}


private:
	float norm = 1.f;

};

class NeuralNetwork
{
public:
	NeuralNetwork(uint3 size, int max_data_num)
	{
		Layers.emplace_back(new InputLayer(size));
		NNOffset.resize(2, 0);
		LayerOffset = {0, Layers.back()->GetMemSize()};
		CUDASize.push_back({dim3(), dim3()});
		N = max_data_num;
		LayerMaxSize = max(LayerMaxSize, Layers.back()->GetMemSize());
	}
	~NeuralNetwork(void)
	{
		checkCuda(cudaFree(DeviceLayerData));
		checkCuda(cudaFree(DeviceNNData));
		checkCuda(cudaFree(DevicedNNData));
		checkCuda(cudaFree(DeviceAnswerData));
		checkCuda(cudaFree(DeviceLayerTem));
	}
	void NNReset(void)
	{
		float* HNN;
		HNN = new float[NNOffset.back()];
		for (size_t i = 0; i < NNOffset.back(); i++) HNN[i] = randN(engine);
		checkCuda(cudaMemcpy(DeviceNNData, HNN, NNOffset.back() * sizeof(float), cudaMemcpyHostToDevice));
	}
	void LayerReset(cudaStream_t stream = cudaStream_t(0))
	{
		checkCuda(cudaMemsetAsync(DeviceLayerData, 0, N * LayerOffset.back() * sizeof(float), stream));
	}
	void SetInput(float* input, int M, cudaStream_t stream = cudaStream_t(0))
	{
		checkCuda(cudaMemcpyAsync(DeviceLayerData, input, M * Layers.front()->GetMemSize() * sizeof(float), cudaMemcpyHostToDevice, stream));
	}
	void SetAnswer(float* answer, int M, cudaStream_t stream = cudaStream_t(0))
	{
		checkCuda(cudaMemcpyAsync(DeviceAnswerData, answer, M * Layers.back()->GetMemSize() * sizeof(float), cudaMemcpyHostToDevice, stream));
	}
	int DeviceMalloc(void)
	{
		checkCuda(cudaMalloc(&DeviceLayerData , N * LayerOffset.back()           * sizeof(float)));
		checkCuda(cudaMalloc(&DeviceNNData    , NNOffset.back()                  * sizeof(float)));
		checkCuda(cudaMalloc(&DevicedNNData   , NNOffset.back()                  * sizeof(float)));
		checkCuda(cudaMalloc(&DeviceAnswerData, N * Layers.back()->GetMemSize()  * sizeof(float)));
		checkCuda(cudaMalloc(&DeviceLayerTem  , N * LayerMaxSize                 * sizeof(float)));
		for (size_t l = 0; l < Layers.size(); l++) {
			DeviceLayerOffset.emplace_back(&DeviceLayerData[N * LayerOffset[l]]);
		}
		return 0;
	}
	void AddLayer(Layer* L, dim3 Block = dim3(), dim3 Thread = dim3())
	{
		
		if (L->CheckLayer(Layers.back()->GetOutSize()) == false) throw ("レイヤー" + to_string(Layers.size()) + "の定義が不適切です").c_str();
		Layers.emplace_back(L);
		NNOffset.emplace_back(NNOffset.back() + L->GetNNSize());
		LayerOffset.emplace_back(LayerOffset.back() + L->GetMemSize());
		CUDASize.push_back({Block, Thread});
		LayerMaxSize = max(LayerMaxSize, L->GetMemSize());
	}
	void run(int M, cudaStream_t stream = cudaStream_t(0))
	{
		assert(M <= N);
		for (size_t l = 1; l < Layers.size(); l++) {
			Layers[l]->run(CUDASize[l][0], CUDASize[l][1], DeviceLayerOffset[l - 1], DeviceLayerOffset[l], &DeviceNNData[NNOffset[l]], N, M, stream);
		}
	}
	void loss(int M, cudaStream_t stream = cudaStream_t(0))
	{
		CheckAns<<<1024, 256, 0, stream>>>(DeviceLayerOffset.back(), DeviceAnswerData, M, Layers.back()->GetMemSize());
	}
	void train(int M, cudaStream_t stream = cudaStream_t(0))
	{
		for (size_t l = Layers.size() - 1; l > 0; l--) {
			Layers[l]->train(DeviceLayerOffset[l - 1], DeviceLayerOffset[l], &DeviceNNData[NNOffset[l]], N, M, &DevicedNNData[NNOffset[l]], stream);
			if (l != 1) {
				Layers[l]->back(CUDASize[l - 1][0], CUDASize[l - 1][1], DeviceLayerOffset[l - 1], DeviceLayerOffset[l], &DeviceNNData[NNOffset[l]], N, M, DeviceLayerTem, stream);
				checkCuda(cudaMemcpyAsync(DeviceLayerOffset[l - 1], DeviceLayerTem, M * Layers[l - 1]->GetMemSize() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
			}
		}
		UpdateWeight<<<1024, 256, 0, stream>>>(DeviceNNData, DevicedNNData, NNOffset.back(), U, H);
	}
	void Inport(string path)
	{
		float* HNN;
		HNN = new float[NNOffset.back()];
		fstream file;
		file.open(path, ios_base::in | ios_base::binary);
		file.read((char*)HNN, NNOffset.back() * sizeof(float));
		file.close();
		checkCuda(cudaMemcpy(DeviceNNData, HNN, NNOffset.back() * sizeof(float), cudaMemcpyHostToDevice));
	}
	void Export(string path)
	{
		float* HNN;
		HNN = new float[NNOffset.back()];
		checkCuda(cudaMemcpy(HNN, DeviceNNData, NNOffset.back() * sizeof(float), cudaMemcpyDeviceToHost));
		fstream file;
		file.open("MyNN.bin", ios_base::out | ios_base::binary);
		file.write((char*)HNN, NNOffset.back() * sizeof(float));
		file.close();
	}

//private:
	vector<shared_ptr<Layer>> Layers;
	vector<int> NNOffset;
	vector<int> LayerOffset;
	vector<float*> DeviceLayerOffset;
	vector<array<dim3, 2>> CUDASize;
	int N;
	int LayerMaxSize = 0;
	float* DeviceLayerData;
	float* DeviceNNData;
	float* DevicedNNData;
	float* DeviceAnswerData;
	float* DeviceLayerTem;

};



float* layer;
/*
float NN1[8][3][3][3][4];
float In [3][32][32];
float Out[8][32][32];
float Cor[8][32][32];//*/

float fun(float x)
{
	return cosf(2 * x);
}

int main()
{

	/*
	const int N = 32;

	NeuralNetwork NN({2, 1, 1}, N);
	NN.AddLayer(new DenseTLayer(2, 4), {1, 1, 32}, {2, 1, 1});
	NN.AddLayer(new DenseTLayer(1, 2), {1, 1, 32}, {1, 1, 1});

	NN.DeviceMalloc();

	NN.NNReset();

	shared_ptr<float[]> input, ans, out;
	input.reset(new float[10000]);
	ans  .reset(new float[10000]);
	out  .reset(new float[10000]);


	float ar[10000];
	for (int e = 0; e < 10000; e++) {

		for (int n = 0; n < 4; n++) {
			int i1 = n / 2;
			int i2 = n % 2;
			input[n * 2] = i1;
			input[n * 2 + 1] = i2;
			ans[n] = i1 ^ i2;
		}


		NN.LayerReset();
		NN.SetInput(input.get(), 4);
		NN.SetAnswer(ans.get(), 4);

		NN.run(4);

		NN.loss(4);
		checkCuda(cudaMemcpy(out.get(), NN.DeviceLayerData, N * NN.LayerOffset.back() * sizeof(float), cudaMemcpyDeviceToHost));
		for (int n = 0; n < N * NN.LayerOffset.back(); n++) {
			ar[n] = out[n];
		}

		NN.train(4);

	}//*/



	/*{
		fstream file;
		cout << "読み込み中...   ";
		file.open("DataSet\\MNIST\\train-labels.idx1-ubyte", ios_base::in | ios_base::binary);
		for (size_t i = 0; i < DATA_NUM; i++) {
			file.read((char*)labels + i, 1);
		}
		file.close();
		file.open("DataSet\\MNIST\\train-images.idx3-ubyte", ios_base::in | ios_base::binary);
		for (size_t i = 0; i < DATA_NUM; i++) {
			file.read((char*)images[i], DATA_SIZE_Y * DATA_SIZE_X);
		}
		file.close();
		cout << "終了" << endl;
	}//*/

	{
		fstream file;
		cout << "読み込み中...   ";
		for (size_t i = 0; i < DATA_NUM; i++) {
			if (i % (DATA_NUM / 5) == 0) {
				file.close();
				file.open("DataSet\\CIFAR-10\\data_batch_" + to_string(i / (DATA_NUM / 5) + 1) + ".bin", ios_base::in | ios_base::binary);
			}
			file.read((char*)labels + i, 1);
			file.read((char*)images + i * DATA_SIZE, DATA_SIZE);
		}
		cout << "終了" << endl;
	}//*/

	
	const int N = Bach;
	const int K = 10;

	//vector<NeuralNetwork> NN;
	
	NeuralNetwork NN({32, 32, 3}, N);
	NN.AddLayer(new ConvolutionLayer(2, 2, 8) , {4, 4, N}, {8, 8, 8});
	NN.AddLayer(new ReluLayer       ()        , {4, 4, N}, {8, 8, 8});
	NN.AddLayer(new APoolingLayer   (2, 2)    , {4, 4, N}, {4, 4, 8});
	NN.AddLayer(new ConvolutionLayer(2, 2, 32), {4, 4, N}, {2, 2, 32});
	NN.AddLayer(new ReluLayer       ()        , {4, 4, N}, {2, 2, 32});
	NN.AddLayer(new APoolingLayer   (2, 2)    , {4, 4, N}, {1, 1, 32});
	NN.AddLayer(new NormalizeLayer  (M_PI)    , {4, 1, N}, {256, 1, 1});
	NN.AddLayer(new DenseTLayer     (256, 16) , {1, 1, N}, {256, 1, 1});
	NN.AddLayer(new BiasLayer       ()        , {1, 1, N}, {256, 1, 1});
	NN.AddLayer(new ReluLayer       ()        , {1, 1, N}, {256, 1, 1});
	NN.AddLayer(new NormalizeLayer  (M_PI)    , {1, 1, N}, {256, 1, 1});
	NN.AddLayer(new DenseTLayer     (10, 32)  , {1, 1, N}, {10, 1, 1});
	NN.AddLayer(new SoftmaxLayer    ()        , {1, 1, N}, {10, 1, 1});

	// 同じ形違う初期値のNNを同時に動かす



	NN.DeviceMalloc();


	//NN.Inport("MyNN.bin");
	NN.NNReset();
	
	shared_ptr<float[]> input, ans, out, nn;
	input.reset(new float[N * NN.Layers.front()->GetMemSize()]);
	ans  .reset(new float[N * NN.Layers.back ()->GetMemSize()]);
	out  .reset(new float[N * NN.Layers.back ()->GetMemSize()]);
	nn   .reset(new float[NN.NNOffset.back()]);

	auto begin = chrono::system_clock::now();

	cudaStream_t stream = cudaStream_t(0);
	//cudaStreamCreate(&stream);

	float preloss = 100.f;

	for (int e = 0; true; e++) {

		auto start = chrono::system_clock::now();

		int loop = (DATA_NUM + Bach - 1) / Bach;

		//printf_s("Epoch %d/%d\n", e + 1, Epoch);
		printf_s("Epoch %d\n", e + 1);

		cout << "0%";

		float loss = 0.f;


		for (int p = 0; p < loop; p++) {

			int M = 0;

			for (int n = 0; n < Bach; n++) {
				int index = p * Bach + n;
				//int index = engine() % DATA_NUM;
				if (index >= DATA_NUM) continue;
				M++;
				for (int k = 0; k < NN.Layers.back()->GetMemSize(); k++) {
					ans.get()[n * 10 + k] = (labels[index] == k ? 1.f : 0.f);
				}
				for (int k = 0; k < DATA_SIZE; k++) {
					input.get()[n * DATA_SIZE + k] = images[index][k] / 255.f;
				}
			}


			cudaStreamSynchronize(stream);

			NN.LayerReset(stream);
			NN.SetInput(input.get(), M, stream);
			NN.SetAnswer(ans.get(), M, stream);

			NN.run(M, stream);
			NN.loss(M, stream);

			checkCuda(cudaMemcpyAsync(out.get(), NN.DeviceLayerOffset.back(), M * NN.Layers.back()->GetMemSize() * sizeof(float), cudaMemcpyDeviceToHost, stream));

			NN.train(M, stream);

			cudaStreamSynchronize(stream);

			for (int i = 0; i < M * NN.Layers.back()->GetMemSize(); i++) {
				loss += abs(out[i]) / 2.f;
				if (_Is_nan(loss)) {
					checkCuda(cudaMemcpyAsync(nn.get(), NN.DeviceNNData, NN.NNOffset.back() * sizeof(float), cudaMemcpyDeviceToHost, stream));
					for (size_t k = 0; k < 1000; k++) {
						cout << nn[k] << endl;
					}
					return -1;
				}
			}

			if(p % 10 == 0) printf("\r%05.1f%%   loss : %010.7f", 100.f * (p + 1) / loop, loss / ((p + 1) * Bach));

		}

		auto end = chrono::system_clock::now();

		//cout << "   " << chrono::duration_cast<chrono::seconds>(end - start).count() << "s" << endl;
		loss /= DATA_NUM;
		printf_s("\r100.0%%   loss : %010.7f  %lds\n", loss, (long)chrono::duration_cast<chrono::seconds>(end - start).count());


		if (end - begin > chrono::minutes(2)) break;
		if (preloss - loss < 0) break;

		preloss = loss;

	}

	cudaStreamSynchronize(stream);

	cudaStreamDestroy(stream);

	NN.Export("MyNN.bin");//*/


	/*for (size_t n = 0; n < 1000; n++) {
		float input[N * DATA_SIZE], ans[N * 10];
		for (size_t i = 0; i < N; i++) {
			for (size_t k = 0; k < DATA_SIZE; k++) {
				input[i * DATA_SIZE + k] = images[i][k] / 255.f;
			}
			for (size_t k = 0; k < 10; k++) {
				ans[i * 10 + k] = (labels[i] == k ? 1.f : 0.f);
			}
		}

		NN.LayerReset();
		NN.SetInput(input);
		NN.SetAnswer(ans);

		NN.run(N);
		float out[N * 10];
		NN.loss(N);

		checkCuda(cudaMemcpy(out, NN.Layers.back()->layer(), N * 10 * sizeof(float), cudaMemcpyDeviceToHost));
		float loss = 0.f;
		for (size_t i = 0; i < N * 10; i++) {
			loss += abs(out[i]);
		}
		cout << loss << endl;


		NN.train(N);


		//cout << "No." << n + 1 << endl;

		
	}
	//*/

	//checkCuda(cudaMemcpy(nn, NN.DeviceNNData, NN.NNOffset.back() * sizeof(float), cudaMemcpyDeviceToHost));


	/*{
		ofstream fileO, fileA;
		fileO.open("AIout.txt");
		fileA.open("Ans.txt");
		for (size_t k = 0; k < 65536; k += N) {
			float input[N];
			for (size_t n = 0; n < N; n++) {
				input[n] = 2 * M_PI * (k + n) / 65536;
			}
			NN.LayerReset();
			NN.SetInput(input);
			NN.run(N);

			float output[N];
			checkCuda(cudaMemcpy(output, NN.Layers.back()->layer(), N * sizeof(float), cudaMemcpyDeviceToHost));

			for (size_t n = 0; n < N; n++) {
				fileO << output[n] << endl;
				fileA << fun(input[n]) << endl;
			}
		}
		fileO.close();
		fileA.close();
	}//*/


	/*cout << "～～確認テスト～～" << endl;
	for (int k = 0; k < 4; k++) {
		double Data[1][1][2] = {{{k / 2, k % 2}}};
		double Ans[1][1][1] = {{{(k / 2) ^ (k % 2)}}};

		double D1[1][1][HNUM];
		for (size_t j = 0; j < HNUM; j++) D1[0][0][j] = 0.0;
		double D2[1][1][HNUM];
		for (size_t j = 0; j < HNUM; j++) D2[0][0][j] = 0.0;
		double D3[1][1][HNUM];
		for (size_t j = 0; j < HNUM; j++) D3[0][0][j] = 0.0;
		double D4[1][1][HNUM];
		for (size_t j = 0; j < HNUM; j++) D4[0][0][j] = 0.0;
		double D5[1][1][1] = {{{0.0}}};

		L1.run(Data, D1);
		L2.run(D1, D2);
		L3.run(D2, D3);
		L4.run(D3, D4);
		L5.run(D4, D5);

		printf_s("%d ^ %d = %d ~ %.3lf\n", k / 2, k % 2, (k / 2) ^ (k % 2), D5[0][0][0]);
	}//*/

	/*cout << "～～関数～～" << endl;
	ofstream file;
	file.open("AIout.txt");
	for (size_t k = 0; k < 10000; k++) {
		double Data[1][1][2] = {{{k / 2, k % 2}}};
		double Ans[1][1][1] = {{{(k / 2) ^ (k % 2)}}};

		double D1[1][1][HNUM];
		for (size_t j = 0; j < HNUM; j++) D1[0][0][j] = 0.0;
		double D2[1][1][HNUM];
		for (size_t j = 0; j < HNUM; j++) D2[0][0][j] = 0.0;
		double D3[1][1][HNUM];
		for (size_t j = 0; j < HNUM; j++) D3[0][0][j] = 0.0;
		double D4[1][1][HNUM];
		for (size_t j = 0; j < HNUM; j++) D4[0][0][j] = 0.0;
		double D5[1][1][1] = {{{0.0}}};

		L1.run(Data, D1);
		L2.run(D1, D2);
		L3.run(D2, D3);
		L4.run(D3, D4);
		L5.run(D4, D5);

		file << D5[0][0][0] << endl;
	}
	file.close();//*/

	/*vector<double> data, zip, redata;
	int N = 20;
	for (size_t i = 0; i < N; i++) {
		data.emplace_back(i);
		//data.emplace_back(engine() % 100);
	}
	for (size_t i = 0; i < N; i++) {
		double a = 0.;
		for (size_t n = 0; n < N; n++) {
			a += data[n] * cos(M_PI / N * (n + 0.5) * i);
		}
		zip.emplace_back(a);
	}
	for (size_t i = 0; i < N; i++) {
		double a = 0.;
		a += zip[0] / 2;
		for (size_t n = 1; n < N - 15; n++) {
			a += zip[n] * cos(M_PI / N * (i + 0.5) * n);
		}
		a *= 2. / N;
		redata.emplace_back(a);
	}
	for (size_t i = 0; i < N; i++) {
		printf_s("%.1lf %.1lf\n", data[i], redata[i]);
	}//*/

	/*fstream file;

	file.open("DataSet\\MNIST\\train-labels.idx1-ubyte", ios_base::in | ios_base::binary);
	for (size_t i = 0; i < DATA_NUM; i++) {
		file.read((char*)labels + i, 1);
	}
	file.close();

	file.open("DataSet\\MNIST\\train-images.idx3-ubyte", ios_base::in | ios_base::binary);
	for (size_t i = 0; i < DATA_NUM; i++) {
		file.read((char*)images[i], DATA_SIZE_Y * DATA_SIZE_X);
		if ((i + 1) % 1000 == 0) cout << i + 1 << endl;
	}
	file.close();//*/

	/*
	for (size_t i = 0; i < DATA_NUM; i++) {
		if (i % (DATA_NUM / 5) == 0) {
			file.close();
			file.open("DataSet\\CIFAR-10\\data_batch_" + to_string(i / (DATA_NUM / 5) + 1) + ".bin", ios_base::in | ios_base::binary);
		}
		file.read((char*)labels + i, 1);
		file.read((char*)images + i * DATA_SIZE, DATA_SIZE);
	}
	file.close();

	float* HNNData = new float[NNSizeAll];
	float* HInputData  = new float[Bach * DATA_SIZE];
	float* HAnswerData = new float[Bach * LayerSize.back()];

	float* DNNData;
	float* DNNDataDif;
	float* DAnswerData;
	float* DLayerData;
	float* DRevTem;

	file.open("MyNN.bin", ios_base::in | ios_base::binary);
	file.read((char*)HNNData, NNSizeAll * sizeof(float));
	file.close();
	
	for (size_t i = 0; i < NNSizeAll; i++) {
		HNNData[i] = randN(engine);
	}

	checkCuda(cudaMalloc(&DNNData      , NNSizeAll            * sizeof(float)));
	checkCuda(cudaMalloc(&DNNDataDif   , NNSizeAll            * sizeof(float)));
	checkCuda(cudaMalloc(&DAnswerData  , Bach * LayerSize.back() * sizeof(float)));
	checkCuda(cudaMalloc(&DLayerData   , Bach * LayerSizeAll     * sizeof(float)));
	checkCuda(cudaMalloc(&DRevTem      , Bach * LayerTemSize     * sizeof(float)));

	checkCuda(cudaMemcpy(DNNData, HNNData, NNSizeAll * sizeof(float), cudaMemcpyHostToDevice));

	checkCuda(cudaMemset(DNNDataDif, 0, NNSizeAll * sizeof(float)));

	//layer = new float[N * LayerSizeAll];
	layer = new float[Bach * LayerSize[7]];


	for (int e = 0; e < Epoch; e++) {

		auto start = chrono::system_clock::now();

		printf_s("Epoch %d/%d\n", e + 1, Epoch);
		for (size_t i = 0; i < (DATA_NUM + Bach - 1) / Bach; i++) {
			cout << "-";
		}
		cout << endl;

		float ok = 0.f;

		for (int p = 0; p < (DATA_NUM + Bach - 1) / Bach; p++) {

			int N = 0;

			for (int n = 0; n < Bach; n++) {
				int index = p * Bach + n;
				if (index >= DATA_NUM) continue;
				N++;
				for (int i = 0; i < LayerSize.back(); i++) {
					HAnswerData[n * LayerSize.back() + i] = (i == labels[index] ? 1.f : 0.f);
				}
				for (int k = 0; k < DATA_SIZE; k++) {
					int x = k % DATA_SIZE_X;
					int y = (k % (DATA_SIZE_X * DATA_SIZE_Y)) / DATA_SIZE_X;
					int z = k / (DATA_SIZE_X * DATA_SIZE_Y);
					HInputData[((n * DATA_SIZE_Z + z) * DATA_SIZE_Y + y) * DATA_SIZE_X + x] = min(1.f, max(0.f, images[index][k] / 255.f + randN(engine) * 0.02f));
				}
			}

			cudaDeviceSynchronize();

			checkCuda(cudaMemset(DAnswerData, 0          , N * LayerSize.back() * sizeof(float)));
			checkCuda(cudaMemset(DLayerData , 0          , N * LayerSizeAll     * sizeof(float)));
			checkCuda(cudaMemcpy(DLayerData , HInputData , N * DATA_SIZE        * sizeof(float), cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(DAnswerData, HAnswerData, N * LayerSize.back() * sizeof(float), cudaMemcpyHostToDevice));
			
			{
				dim3 Block, Thread;

				// 3 * 32 * 32   --0 (IN)
				Block  = {2, 2, 1024};
				Thread = {16, 16, 1};
				ConvolutionT<<<Block, Thread>>>(&DLayerData[N * LayerOfset[0]], &DLayerData[N * LayerOfset[1]], &DNNData[NNOfset[0]], N, 3, 32, 32, 8, 32, 32, 3, 1, 1, 4);
				// 8 * 32 * 32   --1
				Block  = {2, 2, 1024};
				Thread = {8, 8, 4};
				APooling    <<<Block, Thread>>>(&DLayerData[N * LayerOfset[1]], &DLayerData[N * LayerOfset[2]], N, 8, 32, 32, 8, 16, 16, 2, 2);
				// 8 * 16 * 16   --2
				Block  = {1, 1, 1024};
				Thread = {16, 16, 1};
				ConvolutionT<<<Block, Thread>>>(&DLayerData[N * LayerOfset[2]], &DLayerData[N * LayerOfset[3]], &DNNData[NNOfset[1]], N, 8, 16, 16, 16, 16, 16, 8, 2, 2, 8);
				// 16 * 16 * 16  --3
				Block  = {2, 2, 1024};
				Thread = {4, 4, 16};
				APooling    <<<Block, Thread>>>(&DLayerData[N * LayerOfset[3]], &DLayerData[N * LayerOfset[4]], N, 16, 16, 16, 16, 8, 8, 2, 2);
				// 16 * 8 * 8    --4
				Block  = {16, 1, 1024};
				Thread = {16, 1, 1};
				EachDenseT  <<<Block, Thread>>>(&DLayerData[N * LayerOfset[4]], &DLayerData[N * LayerOfset[5]], &DNNData[NNOfset[2]], N, 16, 8, 8, 16, 16);
				// 256           --5
				Block  = {1 , 1, 1024};
				Thread = {10, 1, 1};
				DenseT      <<<Block, Thread>>>(&DLayerData[N * LayerOfset[5]], &DLayerData[N * LayerOfset[6]], &DNNData[NNOfset[3]], N, 1, 1, 256, 10, 32);
				// 10            --6
				Block  = {1 , 1, 1024};
				Thread = {10, 1, 16};
				Softmax     <<<Block, Thread>>>(&DLayerData[N * LayerOfset[6]], &DLayerData[N * LayerOfset[7]], N, 10, 10);
				// 10            --7 (OUT)

				checkCuda(cudaMemcpy(layer, &DLayerData[N * LayerOfset[7]], N* LayerSize[7] * sizeof(float), cudaMemcpyDeviceToHost));
				
				Block  = {1024 , 1, 1};
				Thread = {256, 1, 1};
				// 逆伝播
				CheckAns   <<<Block, Thread>>>(&DLayerData[N * LayerOfset[7]], DAnswerData, N ,LayerSize.back());

				// 10            --7 (In)
				Block  = {1 , 1, 1024};
				Thread = {10, 1, 16};
				RevSoftmaxL     <<<Block, Thread>>>(DRevTem                , &DLayerData[N * LayerOfset[6]], &DLayerData[N * LayerOfset[7]], N, 10, 10);
				checkCuda(cudaMemcpy(&DLayerData[N * LayerOfset[6]], DRevTem, N * LayerSize[6] * sizeof(float), cudaMemcpyDeviceToDevice));
				// 10            --6
				Block  = {256 , 1, 1};
				Thread = {10, 32, 1};
				RevDenseTN      <<<Block, Thread>>>(&DNNDataDif[NNOfset[3]], &DLayerData[N * LayerOfset[5]], &DLayerData[N * LayerOfset[6]], &DNNData[NNOfset[3]], N, 1, 1, 256, 10, 32);
				Block  = {1 , 1, 1024};
				Thread = {256, 1, 1};
				RevDenseTL      <<<Block, Thread>>>(DRevTem                , &DLayerData[N * LayerOfset[5]], &DLayerData[N * LayerOfset[6]], &DNNData[NNOfset[3]], N, 1, 1, 256, 10, 32);
				checkCuda(cudaMemcpy(&DLayerData[N * LayerOfset[5]], DRevTem, N * LayerSize[5] * sizeof(float), cudaMemcpyDeviceToDevice));
				// 256           --5
				Block  = {8 , 8, 16};
				Thread = {16, 16, 1};
				RevEachDenseTN  <<<Block, Thread>>>(&DNNDataDif[NNOfset[2]], &DLayerData[N * LayerOfset[4]], &DLayerData[N * LayerOfset[5]], &DNNData[NNOfset[2]], N, 16, 8, 8, 16, 16);
				Block  = {1 , 1, 1024};
				Thread = {8, 8, 4};
				RevEachDenseTL  <<<Block, Thread>>>(DRevTem                , &DLayerData[N * LayerOfset[4]], &DLayerData[N * LayerOfset[5]], &DNNData[NNOfset[2]], N, 16, 8, 8, 16, 16);
				checkCuda(cudaMemcpy(&DLayerData[N * LayerOfset[4]], DRevTem, N * LayerSize[4] * sizeof(float), cudaMemcpyDeviceToDevice));
				// 16 * 8 * 8    --4
				Block  = {1 , 1, 1024};
				Thread = {8, 8, 4};
				RevAPoolingL    <<<Block, Thread>>>(DRevTem                , &DLayerData[N * LayerOfset[3]], &DLayerData[N * LayerOfset[4]], N, 16, 16, 16, 16, 8, 8, 2, 2);
				checkCuda(cudaMemcpy(&DLayerData[N * LayerOfset[3]], DRevTem, N * LayerSize[3] * sizeof(float), cudaMemcpyDeviceToDevice));
				// 16 * 16 * 16  --3
				Block  = {5 , 5, 8};
				Thread = {16, 8, 1};
				RevConvolutionTN<<<Block, Thread>>>(&DNNDataDif[NNOfset[1]], &DLayerData[N * LayerOfset[2]], &DLayerData[N * LayerOfset[3]], &DNNData[NNOfset[1]], N, 8, 16, 16, 16, 16, 16, 8, 2, 2, 8);
				Block  = {1 , 1, 1024};
				Thread = {16, 16, 1};
				RevConvolutionTL<<<Block, Thread>>>(DRevTem                , &DLayerData[N * LayerOfset[2]], &DLayerData[N * LayerOfset[3]], &DNNData[NNOfset[1]], N, 8, 16, 16, 16, 16, 16, 8, 2, 2, 8);
				checkCuda(cudaMemcpy(&DLayerData[N * LayerOfset[2]], DRevTem, N * LayerSize[2] * sizeof(float), cudaMemcpyDeviceToDevice));
				// 8 * 16 * 16   --2
				Block  = {1 , 1, 1024};
				Thread = {16, 16, 1};
				RevAPoolingL    <<<Block, Thread>>>(DRevTem                , &DLayerData[N * LayerOfset[1]], &DLayerData[N * LayerOfset[2]], N, 8, 32, 32, 8, 16, 16, 2, 2);
				checkCuda(cudaMemcpy(&DLayerData[N * LayerOfset[1]], DRevTem, N * LayerSize[1] * sizeof(float), cudaMemcpyDeviceToDevice));
				// 8 * 32 * 32   --1
				Block  = {3 , 3, 3};
				Thread = {8, 4, 1};
				RevConvolutionTN<<<Block, Thread>>>(&DNNDataDif[NNOfset[0]], &DLayerData[N * LayerOfset[0]], &DLayerData[N * LayerOfset[1]], &DNNData[NNOfset[0]], N, 3, 32, 32, 8, 32, 32, 3, 1, 1, 4);
				// 3 * 32 * 32   --0 (Out)

				/// 最初の畳み込みのRevConvolutionNが一番遅い

				// 更新
				Block  = {1024 , 1, 1};
				Thread = {256, 1, 1};
				UpdateWeight<<<Block, Thread>>>(DNNData, DNNDataDif, NNSizeAll, U, H);

			}


			for (int n = 0; n < N; n++) {
				int out = 0;
				float ot = 0.f;
				int ans = 0;
				float at = 0.f;
				for (int i = 0; i < 10; i++) {
					if (layer[n * 10 + i] > ot) {
						ot = layer[n * 10 + i];
						out = i;
					}
					if (HAnswerData[n * 10 + i] > at) {
						at = HAnswerData[n * 10 + i];
						ans = i;
					}
				}
				if (ans == out) ok++;
			}

			cout << ".";

		}

		auto end = chrono::system_clock::now();

		cout << "   " << chrono::duration_cast<chrono::seconds>(end - start).count() << "s" << endl;

		printf_s("正解率 : %04.1f%%\n", 100.0f * ok / DATA_NUM);


		/*
		cout << "L0" << endl;
		for (size_t n = 0; n < N; n++) {
			cout << "No." << n + 1 << endl;
			for (size_t z = 0; z < 3; z++) {
				for (size_t y = 0; y < 32; y++) {
					for (size_t x = 0; x < 32; x++) {
						printf_s("%01.2f ", layer[N * LayerOfset[0] + ((n * 3 + z) * 32 + y) * 32 + x]);
					}
					cout << endl;
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << "L1" << endl;
		for (size_t n = 0; n < N; n++) {
			cout << "No." << n + 1 << endl;
			for (size_t z = 0; z < 8; z++) {
				for (size_t y = 0; y < 32; y++) {
					for (size_t x = 0; x < 32; x++) {
						printf_s("%03.0f ", layer[N * LayerOfset[1] + ((n * 8 + z) * 32 + y) * 32 + x]);
					}
					cout << endl;
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << "L2" << endl;
		for (size_t n = 0; n < N; n++) {
			cout << "No." << n + 1 << endl;
			for (size_t z = 0; z < 8; z++) {
				for (size_t y = 0; y < 16; y++) {
					for (size_t x = 0; x < 16; x++) {
						printf_s("%04.0f ", layer[N * LayerOfset[2] + ((n * 8 + z) * 16 + y) * 16 + x]);
					}
					cout << endl;
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << "L3" << endl;
		for (size_t n = 0; n < N; n++) {
			cout << "No." << n + 1 << endl;
			for (size_t z = 0; z < 16; z++) {
				for (size_t y = 0; y < 16; y++) {
					for (size_t x = 0; x < 16; x++) {
						printf_s("%03.0f ", layer[N * LayerOfset[3] + ((n * 16 + z) * 16 + y) * 16 + x]);
					}
					cout << endl;
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << "L4" << endl;
		for (size_t n = 0; n < N; n++) {
			cout << "No." << n + 1 << endl;
			for (size_t z = 0; z < 16; z++) {
				for (size_t y = 0; y < 8; y++) {
					for (size_t x = 0; x < 8; x++) {
						printf_s("%03.0f ", layer[N * LayerOfset[4] + ((n * 16 + z) * 8 + y) * 8 + x]);
					}
					cout << endl;
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << "L5" << endl;
		for (size_t n = 0; n < N; n++) {
			cout << "No." << n + 1 << endl;
			for (size_t x1 = 0; x1 < 16; x1++) {
				for (size_t x2 = 0; x2 < 16; x2++) {
					printf_s("%03.0f ", layer[N * LayerOfset[5] + (n * 16 + x1) * 16 + x2]);
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << "L6" << endl;
		for (size_t n = 0; n < N; n++) {
			cout << "No." << n + 1 << endl;
			for (size_t x = 0; x < 10; x++) {
				printf_s("%04.0f ", layer[N * LayerOfset[6] + n * 10 + x]);
			}
			cout << endl;
		}
		cout << "L7" << endl;
		for (size_t n = 0; n < N; n++) {
			cout << "No." << n + 1 << endl;
			for (size_t x = 0; x < 10; x++) {
				printf_s("%.4f ", layer[N * LayerOfset[7] + n * 10 + x]);
			}
			cout << endl;
		}
	}
	
	checkCuda(cudaMemcpy(HNNData, DNNData, NNSizeAll * sizeof(float), cudaMemcpyDeviceToHost));

	file.open("MyNN.bin", ios_base::out | ios_base::binary);
	file.write((char*)HNNData, NNSizeAll * sizeof(float));
	file.close();


	delete[] HNNData;
	delete[] HInputData;
	delete[] HAnswerData;

	cudaFree(DNNData);
	cudaFree(DNNDataDif);
	cudaFree(DAnswerData);
	cudaFree(DLayerData);
	cudaFree(DRevTem);
	//*/

	return 0;
}

