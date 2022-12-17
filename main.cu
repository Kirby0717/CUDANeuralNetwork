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
#include <algorithm>
#include <thread>
#include <chrono>
#include <stdio.h>
#include <assert.h>

using namespace std;
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

random_device seed_gen;
mt19937_64 engine(seed_gen());
normal_distribution<float> randN(0.f, 0.05f);
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
	if (thz >= olz || oy >= oly || ox >= olx) return;
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
	if (fz >= flz || fy >= fly || fx >= flx) return;
	int Iindex;
	int Oindex;
	int Nindex;
	float sum = 0.f;
	Nindex = ((oz * flz + fz) * fly + fy) * flx + fx;
	for (int dn = 0; dn < n; dn++) {
		for (int iy = max(0, fry - fy); iy < ily + min(0, fry - fy); iy++)
		for (int ix = max(0, frx - fx); ix < ilx + min(0, frx - fx); ix++) {
			Iindex = ((dn * ilz + fz) * ily + iy) * ilx + ix;
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
	if (thz >= olz || iy >= oly || ix >= olx) return;
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
/*
* f* := プーリングの*方向の大きさ
* il* := 入力データの*方向の大きさ
* ol* := 出力データの*方向の大きさ
* n := 計算するデータ数
*/
__global__ void APooling(const float* In, float* Out, int n, int ilz, int ily, int ilx, int olz, int oly, int olx, int fy, int fx)
{
	const int thz =                          threadIdx.z;
	const int oy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ox = blockDim.x * blockIdx.x + threadIdx.x;
	if (thz >= olz || oy >= oly || ox >= olx) return;
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
	// ブロックサイズとスレッドサイズは入力と同じだが、出力が入力より小さいという定義を使って、出力と同じサイズだけ動かす
	const int thz =                          threadIdx.z;
	const int oy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ox = blockDim.x * blockIdx.x + threadIdx.x;
	if (thz >= olz || oy >= oly || ox >= olx) return;
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
	if (thz >= olz || oy >= oly || ox >= olx) return;
	int Iindex;
	int Oindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		for (int ioz = thz; ioz < olz; ioz += blockDim.z) {
			Oindex = ((dn * olz + ioz) * oly + oy) * olx + ox;
			Out[Oindex] = -FLT_MAX;
			for (int dy = 0; dy < fy; dy++)
			for (int dx = 0; dx < fx; dx++) {
				Iindex = ((dn * ilz + ioz) * ily + oy * fy + dy) * ilx + ox * fx + dx;
				Out[Oindex] = max(Out[Oindex], In[Iindex]);
			}
		}
	}
	return;
}
__global__ void RevMPoolingL(float* Tem, const float* In, const float* Out, int n, int ilz, int ily, int ilx, int olz, int oly, int olx, int fy, int fx)
{
	// ブロックサイズとスレッドサイズは入力と同じだが、出力が入力より小さいという定義を使って、出力と同じサイズだけ動かす
	const int thz =                          threadIdx.z;
	const int oy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ox = blockDim.x * blockIdx.x + threadIdx.x;
	if (thz >= olz || oy >= oly || ox >= olx) return;
	int Iindex;
	int Oindex;
	int Mindex = 0;
	float maxv = -FLT_MAX;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		for (int ioz = thz; ioz < olz; ioz += blockDim.z) {
			Oindex = ((dn * olz + ioz) * oly + oy) * olx + ox;
			for (int dy = 0; dy < fy; dy++)
			for (int dx = 0; dx < fx; dx++) {
				Iindex = ((dn * ilz + ioz) * ily + oy * fy + dy) * ilx + ox * fx + dx;
				if (In[Iindex] > maxv) {
					Tem[Mindex] = 0.f;
					Tem[Iindex] = Out[Oindex];
					Mindex = Iindex;
					maxv = In[Iindex];
				}
				else {
					Tem[Iindex] = 0.f;
				}
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
	if (iz >= ilz || ox >= olb) return;
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
	if (iz >= ilz || iy >= ily || ix >= ilx || ox >= olb || di >= d) return;
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
	const int thz =                          threadIdx.z;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	if (thz >= ilz || iy >= ily || ix >= ilx) return;
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
	if (ox >= olx) return;
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
	if (iz >= ilz || iy >= ily || ix >= ilx || ox >= olx) return;
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
	const int thz =                          threadIdx.z;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	if (thz >= ilz || iy >= ily || ix >= ilx) return;
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
	if (ox >= olx) return;
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
	if (iz >= ilz || iy >= ily || ix >= ilx || ox >= olx || di >= d) return;
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
	const int thz =                          threadIdx.z;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	if (thz >= ilz || iy >= ily || ix >= ilx) return;
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
	}
	return;
}
/*
* n := 計算するデータ数
*/
__global__ void Softmax(const float* In, float* Out, int n, int ilx, int olx)
{
	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= olx) return;
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
	if (x >= ilx) return;
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
	if (z >= lz || y >= ly || x >= lx) return;
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
	if (z >= lz || y >= ly || x >= lx) return;
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
	if (z >= lz || y >= ly || x >= lx) return;
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
	if (z >= lz || y >= ly || x >= lx) return;
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
	if (z >= lz || y >= ly || x >= lx) return;
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
__global__ void Limit(const float* In, float* Out, int n, int lz, int ly, int lx, float lmin, float lmax)
{
	const int z = threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (z >= lz || y >= ly || x >= lx) return;
	int IOindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		IOindex = ((dn * lz + z) * ly + y) * lx + x;
		Out[IOindex] = min(lmax, max(lmin, In[IOindex]));
	}
	return;
}
__global__ void RevLimitL(float* Tem, const float* In, float* Out, int n, int lz, int ly, int lx, float lmin, float lmax)
{
	const int z = threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (z >= lz || y >= ly || x >= lx) return;
	int IOindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		IOindex = ((dn * lz + z) * ly + y) * lx + x;
		if (lmin < In[IOindex] && In[IOindex] < lmax) Tem[IOindex] = Out[IOindex];
		else                                          Tem[IOindex] = 0.f;
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
			tem[Index] = max(tem[Index], abs(In[dn * stride + i]));
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
* l* := 入出力データの*方向の大きさ
*/
__global__ void Dropout(const float* In, float* Out, int n, int lz, int ly, int lx, float* DropFlag)
{
	const int z =                           threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (z >= lz || y >= ly || x >= lx) return;
	int IOindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		IOindex = ((dn * lz + z) * ly + y) * lx + x;
		Out[IOindex] = DropFlag[IOindex] * In[IOindex];
	}
	return;
}
__global__ void RevDropoutL(float* Tem, const float* In, float* Out, int n, int lz, int ly, int lx, float* DropFlag, float scale)
{
	const int z =                           threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (z >= lz || y >= ly || x >= lx) return;
	int IOindex;
	for (int dn = blockIdx.z; dn < n; dn += gridDim.z) {
		IOindex = ((dn * lz + z) * ly + y) * lx + x;
		if (DropFlag[IOindex] > 0.5f) Tem[IOindex] = Out[IOindex];
		else                          Tem[IOindex] = 0.f;
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
__global__ void UpdateWeight_Adam(float* NN, int n, int t, float* dNN, float* m, float* v)
{
	const int Index = blockDim.x * blockIdx.x + threadIdx.x;
	const float Alpha = 0.001f, Beta1 = 0.9f, Beta2 = 0.999f, Epsilon = 1e-7f;
	float tm, tv;
	for (int dn = Index; dn < n; dn += gridDim.x * blockDim.x) {
		m[dn] = Beta1 * m[dn] + (1 - Beta1) * dNN[dn];
		v[dn] = Beta2 * v[dn] + (1 - Beta2) * dNN[dn] * dNN[dn];
		tm = m[dn] / (1 - pow(Beta1, t));
		tv = v[dn] / (1 - pow(Beta2, t));
		NN[dn] -= (Alpha * tm) / (sqrt(tv) + Epsilon);
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
	virtual int GetMemSizeL(void) { return OutSize.z * OutSize.y * OutSize.x; }
	virtual bool CheckLayer(uint3 in_size) { return false; }
	virtual int GetNNSize(void) const { return -1; }
	virtual void SetLayer(float* layer, int N) { return; }
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
class LimitLayer : public Layer
{
public:
	LimitLayer(float lmin, float lmax) : lmin(lmin), lmax(lmax) { }
	~LimitLayer(void) = default;
	bool CheckLayer(uint3 in_size) override { OutSize = InSize = in_size; return true; }
	int GetNNSize(void) const override { return 0; }
	void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) override
	{
		Limit <<<Block, Thread, 0, stream>>>(in_layer, out_layer, M, InSize.z, InSize.y, InSize.x, lmin, lmax);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override { }
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevLimitL<<<Block, Thread, 0, stream>>>(tem, in_layer, out_layer, M, InSize.z, InSize.y, InSize.x, lmin, lmax);
	}

private:
	float lmin, lmax;

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
	int GetNNSize(void) const override { return num * InSize.z * (2 * fry + 1) * (2 * frx + 1); }
	void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) override
	{
		Convolution<<<Block, Thread, 0, stream>>>(in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x, OutSize.z, OutSize.y, OutSize.x, InSize.z, fry, frx);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override
	{
		dim3 Block = {2 * frx + 1, 2 * fry + 1, InSize.z}, Thread = {num, 1, 1};
		RevConvolutionN<<<Block, Thread, 0, stream>>>(dnn, in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x, OutSize.z, OutSize.y, OutSize.x, InSize.z, fry, frx);
	}
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevConvolutionL<<<Block, Thread, 0, stream>>>(tem, in_layer, out_layer, nn, M, InSize.z, InSize.y, InSize.x, OutSize.z, OutSize.y, OutSize.x, InSize.z, fry, frx);
	}

private:
	unsigned int fry, frx, num;

};
class MPoolingLayer : public Layer
{
public:
	MPoolingLayer(int fx, int fy) : fx(fx), fy(fy) { }
	~MPoolingLayer(void) = default;
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
		MPooling <<<Block, Thread, 0, stream>>>(in_layer, out_layer, M, InSize.z, InSize.y, InSize.x, OutSize.z, OutSize.y, OutSize.x, fy, fx);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override { }
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevMPoolingL<<<Block, Thread, 0, stream>>>(tem, in_layer, out_layer, M, InSize.z, InSize.y, InSize.x, OutSize.z, OutSize.y, OutSize.x, fy, fx);
	}

private:
	int fx, fy;

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
		Normalize<<<Block, Thread, 0, stream>>>(in_layer, out_layer, M, InSize.z, InSize.y, InSize.x, norm, &out_layer[N * OutSize.z * OutSize.y * OutSize.x]);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override { }
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevNormalizeL<<<Block, Thread, 0, stream>>>(tem, in_layer, out_layer, M, InSize.z, InSize.y, InSize.x, norm, &out_layer[N * OutSize.z * OutSize.y * OutSize.x]);
	}


private:
	float norm = 1.f;

};
class DropoutLayer : public Layer
{
public:
	DropoutLayer(float p = 0.5f) : p(p) { }
	~DropoutLayer(void) = default;
	bool CheckLayer(uint3 in_size) override { OutSize = InSize = in_size; return true; }
	int GetNNSize(void) const override { return 0; }
	int GetMemSize(void) override { return 2 * OutSize.z * OutSize.y * OutSize.x; }
	void SetLayer(float* layer, int N) override
	{
		float* HNN;
		HNN = new float[N * OutSize.z * OutSize.y * OutSize.x];
		for (size_t i = 0; i < N * OutSize.z * OutSize.y * OutSize.x; i++) HNN[i] = (randU(engine) < p ? 0.f : 1.f);
		checkCuda(cudaMemcpy(&layer[N * OutSize.z * OutSize.y * OutSize.x], HNN, N * OutSize.z * OutSize.y * OutSize.x * sizeof(float), cudaMemcpyHostToDevice));
		delete[] HNN;
	}
	void run  (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, cudaStream_t stream = cudaStream_t(0)) override
	{
		Dropout <<<Block, Thread, 0, stream>>>(in_layer, out_layer, M, InSize.z, InSize.y, InSize.x, &out_layer[N * OutSize.z * OutSize.y * OutSize.x]);
	}
	void train(                         float* in_layer, float* out_layer, float* nn, int N, int M, float* dnn, cudaStream_t stream = cudaStream_t(0)) override { }
	void back (dim3 Block, dim3 Thread, float* in_layer, float* out_layer, float* nn, int N, int M, float* tem, cudaStream_t stream = cudaStream_t(0)) override
	{
		RevDropoutL <<<Block, Thread, 0, stream>>>(tem, in_layer, out_layer, M, InSize.z, InSize.y, InSize.x, &out_layer[N * OutSize.z * OutSize.y * OutSize.x], 1 / (1 - p));
	}

private:
	float p = 0.5f; //無効化する確率
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
		checkCuda(cudaFree(DeviceNNDataM));
		checkCuda(cudaFree(DeviceNNDataV));
		checkCuda(cudaFree(DeviceAnswerData));
		checkCuda(cudaFree(DeviceLayerTem));
	}
	void NNReset(void)
	{
		float* HNN;
		HNN = new float[NNOffset.back()];
		for (size_t i = 0; i < NNOffset.back(); i++) HNN[i] = randN(engine);
		checkCuda(cudaMemcpy(DeviceNNData, HNN, NNOffset.back() * sizeof(float), cudaMemcpyHostToDevice));
		delete[] HNN;
	}
	void LayerReset(cudaStream_t stream = cudaStream_t(0))
	{
		checkCuda(cudaMemsetAsync(DeviceLayerData, 0, N * LayerOffset.back() * sizeof(float), stream));
		for (size_t l = 1; l < Layers.size(); l++) {
			Layers[l]->SetLayer(DeviceLayerOffset[l], N);
		}
	}
	void NNUpdateDataReset(void)
	{
		checkCuda(cudaMemsetAsync(DeviceNNDataM, 0, NNOffset.back() * sizeof(float)));
		checkCuda(cudaMemsetAsync(DeviceNNDataV, 0, NNOffset.back() * sizeof(float)));
	}
	void SetInput(float* input, int M, cudaStream_t stream = cudaStream_t(0))
	{
		checkCuda(cudaMemcpyAsync(DeviceLayerData, input, M * Layers.front()->GetMemSize() * sizeof(float), cudaMemcpyHostToDevice, stream));
	}
	void SetAnswer(float* answer, int M, cudaStream_t stream = cudaStream_t(0))
	{
		checkCuda(cudaMemcpyAsync(DeviceAnswerData, answer, M * Layers.back()->GetMemSize() * sizeof(float), cudaMemcpyHostToDevice, stream));
	}
	void TrainSetUp(float* input, float* answer, int M, cudaStream_t stream = cudaStream_t(0))
	{
		this->LayerReset(stream);
		this->SetInput(input, M, stream);
		this->SetAnswer(answer, M, stream);
	}
	int DeviceMalloc(void)
	{
		checkCuda(cudaMalloc(&DeviceLayerData , N * LayerOffset.back()           * sizeof(float)));
		checkCuda(cudaMalloc(&DeviceNNData    , NNOffset.back()                  * sizeof(float)));
		checkCuda(cudaMalloc(&DevicedNNData   , NNOffset.back()                  * sizeof(float)));
		checkCuda(cudaMalloc(&DeviceNNDataM   , NNOffset.back()                  * sizeof(float)));
		checkCuda(cudaMalloc(&DeviceNNDataV   , NNOffset.back()                  * sizeof(float)));
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
	void out(float* out, int M, cudaStream_t stream = cudaStream_t(0))
	{
		checkCuda(cudaMemcpyAsync(out, DeviceLayerOffset.back(), M * Layers.back()->GetMemSize() * sizeof(float), cudaMemcpyDeviceToHost, stream));
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
				checkCuda(cudaMemcpyAsync(DeviceLayerOffset[l - 1], DeviceLayerTem, M * Layers[l - 1]->GetMemSizeL() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
			}
		}
	}
	void update(int t, cudaStream_t stream = cudaStream_t(0))
	{
		UpdateWeight_Adam<<<1024, 256, 0, stream>>>(DeviceNNData, NNOffset.back(), t, DevicedNNData, DeviceNNDataM, DeviceNNDataV);
	}
	int fit(vector<float> images, vector<int> labels, int Epoch, int Bach = 32)
	{
		cudaStream_t stream = cudaStream_t(0);
		const int data_num = labels.size();
		const dim3 InputSize = this->Layers.front()->GetOutSize();
		const dim3 OutputSize = this->Layers.back()->GetOutSize();
		const int input_size = InputSize.x * InputSize.y * InputSize.z;
		const int output_size = OutputSize.x * OutputSize.y * OutputSize.z;
		const int loop = (data_num + Bach - 1) / Bach;
		int ucount = 1;

		if (images.size() != data_num * input_size) return -1;

		shared_ptr<float[]> input, ans, out;

		input.reset(new float[Bach * input_size ]);
		ans  .reset(new float[Bach * output_size]);
		out  .reset(new float[Bach * output_size]);

		this->NNUpdateDataReset();

		for (int e = 0; e < Epoch; e++) {

			auto start = chrono::system_clock::now();

			int lcount = 0;
			int acount = 0;
			float loss = 0.f;
			float accuracy = 0.f;

			printf_s("Epoch %d\n", e + 1);

			vector<int> SIndex;
			for (int i = 0; i < data_num; i++) SIndex.emplace_back(i);
			shuffle(SIndex.begin(), SIndex.end(), engine);

			for (int p = 0; p < loop; p++) {

				int M = 0;

				for (int n = 0; n < Bach; n++) {
					int index = p * Bach + n;
					if (index >= data_num) continue;
					index = SIndex[index];
					M++;
					for (int k = 0; k < output_size; k++) {
						ans.get()[n * 10 + k] = (labels[index] == k ? 1.f : 0.f);
					}
					for (int k = 0; k < input_size; k++) {
						input.get()[n * input_size + k] = images[index * input_size + k];
					}
				}


				cudaStreamSynchronize(stream);

				this->LayerReset(stream);
				this->SetInput(input.get(), M, stream);
				this->SetAnswer(ans.get(), M, stream);
				this->run(M, stream);
				this->loss(M, stream);
				this->out(out.get(), M);
				this->train(M, stream);
				this->update(ucount++, stream);

				for (int m = 0; m < M; m++) {
					float ev = 1.f + out[m * this->Layers.back()->GetMemSize() + labels[SIndex[p * Bach + m]]];
					bool ok = true;
					for (int i = 0; i < this->Layers.back()->GetMemSize(); i++) {
						if (i != labels[SIndex[p * Bach + m]] && out[m * this->Layers.back()->GetMemSize() + i] > ev) ok = false;
						loss += out[m * this->Layers.back()->GetMemSize() + i] * out[m * this->Layers.back()->GetMemSize() + i];
						lcount++;
						if (_Is_nan(loss) || _Is_inf(loss)) {
							cudaStreamSynchronize(stream);
							shared_ptr<float[]> nn;
							nn.reset(new float[this->NNOffset.back()]);
							checkCuda(cudaMemcpyAsync(nn.get(), this->DeviceNNData, this->NNOffset.back() * sizeof(float), cudaMemcpyDeviceToHost, stream));
							for (size_t k = 0; k < 100; k++) {
								cout << nn[k] << endl;
							}
							return -1;
						}
					}
					acount++;
					if (ok) accuracy += 1.f;
				}

				if (p % (1024 / Bach) == 0) printf("\r%05.1f%%   loss : %010.7f   accuracy : %010.7f", 100.f * (p + 1) / loop, loss / lcount, accuracy / acount);

			}

			auto end = chrono::system_clock::now();

			printf_s("\r100.0%%   loss : %010.7f   accuracy : %010.7f  %.3lfs\n", loss / lcount, accuracy / acount, (long)chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.0);

		}

		cudaStreamSynchronize(stream);

		cudaStreamDestroy(stream);

		return 0;
	}
	int test(vector<float> images, vector<int> labels, int Bach = 32)
	{
		cudaStream_t stream = cudaStream_t(0);
		const int data_num = labels.size();
		const dim3 InputSize = this->Layers.front()->GetOutSize();
		const dim3 OutputSize = this->Layers.back()->GetOutSize();
		const int input_size = InputSize.x * InputSize.y * InputSize.z;
		const int output_size = OutputSize.x * OutputSize.y * OutputSize.z;
		const int loop = (data_num + Bach - 1) / Bach;

		if (images.size() != data_num * input_size) return -1;

		shared_ptr<float[]> input, ans, out;

		input.reset(new float[Bach * input_size ]);
		ans  .reset(new float[Bach * output_size]);
		out  .reset(new float[Bach * output_size]);


		auto start = chrono::system_clock::now();

		int lcount = 0;
		int acount = 0;
		float loss = 0.f;
		float accuracy = 0.f;

		cout << "Test" << endl;

		for (int p = 0; p < loop; p++) {

			int M = 0;

			for (int n = 0; n < Bach; n++) {
				int index = p * Bach + n;
				if (index >= data_num) continue;
				M++;
				for (int k = 0; k < output_size; k++) {
					ans.get()[n * 10 + k] = (labels[index] == k ? 1.f : 0.f);
				}
				for (int k = 0; k < input_size; k++) {
					input.get()[n * input_size + k] = images[index * input_size + k];
				}
			}


			cudaStreamSynchronize(stream);

			this->LayerReset(stream);
			this->SetInput(input.get(), M, stream);
			this->SetAnswer(ans.get(), M, stream);
			this->run(M, stream);
			this->loss(M, stream);
			this->out(out.get(), M);

			for (int m = 0; m < M; m++) {
				float ev = 1.f + out[m * this->Layers.back()->GetMemSize() + labels[p * Bach + m]];
				bool ok = true;
				for (int i = 0; i < this->Layers.back()->GetMemSize(); i++) {
					if (i != labels[p * Bach + m] && out[m * this->Layers.back()->GetMemSize() + i] > ev) ok = false;
					loss += out[m * this->Layers.back()->GetMemSize() + i] * out[m * this->Layers.back()->GetMemSize() + i];
					lcount++;
					if (_Is_nan(loss) || _Is_inf(loss)) {
						cudaStreamSynchronize(stream);
						shared_ptr<float[]> nn;
						checkCuda(cudaMemcpyAsync(nn.get(), this->DeviceNNData, this->NNOffset.back() * sizeof(float), cudaMemcpyDeviceToHost, stream));
						for (size_t k = 0; k < 1000; k++) {
							cout << nn[k] << endl;
						}
						return -1;
					}
				}
				acount++;
				if (ok) accuracy += 1.f;
			}

			if (p % (1024 / Bach) == 0) printf("\r%05.1f%%   loss : %010.7f   accuracy : %010.7f", 100.f * (p + 1) / loop, loss / lcount, accuracy / acount);

		}

		auto end = chrono::system_clock::now();

		printf_s("\r100.0%%   loss : %010.7f   accuracy : %010.7f  %.3lfs\n", loss / lcount, accuracy / acount, (long)chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.0);


		cudaStreamSynchronize(stream);

		cudaStreamDestroy(stream);

		return 0;
	}
	vector<float> infer(vector<float> images)
	{
		cudaStream_t stream = cudaStream_t(0);
		const dim3 InputSize = this->Layers.front()->GetOutSize();
		const dim3 OutputSize = this->Layers.back()->GetOutSize();
		const int input_size = InputSize.x * InputSize.y * InputSize.z;
		const int output_size = OutputSize.x * OutputSize.y * OutputSize.z;

		if (images.size() != input_size) return vector<float>();

		shared_ptr<float[]> input, out;

		input.reset(new float[input_size]);
		out  .reset(new float[output_size]);

		for (int i = 0; i < input_size; i++) input.get()[i] = images[i];

		this->LayerReset(stream);
		this->SetInput(input.get(), 1, stream);
		this->run(1, stream);
		this->out(out.get(), 1, stream);

		cudaStreamSynchronize(stream);

		cudaStreamDestroy(stream);

		vector<float> output;
		for (int i = 0; i < input_size; i++) output.emplace_back(out[i]);

		return output;
	}
	void Import(string path)
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
	float* DeviceNNDataM;
	float* DeviceNNDataV;
	float* DeviceAnswerData;
	float* DeviceLayerTem;

};



float fun(float x)
{
	return cosf(2 * x);
}

int main()
{



	vector<float> train_images, test_images;
	vector<int> train_labels, test_labels;

	{
		fstream file;
		char tem[28 * 28];
		cout << "読み込み中...   ";

		file.open("DataSet\\MNIST\\train-images.idx3-ubyte", ios_base::in | ios_base::binary);
		train_images.reserve(60000 * 28 * 28);
		for (size_t i = 0; i < 60000; i++) {
			file.read(tem, 28 * 28);
			for (size_t j = 0; j < 28 * 28; j++) train_images.emplace_back((unsigned char)tem[j] / 255.f);
		}
		file.close();

		file.open("DataSet\\MNIST\\train-labels.idx1-ubyte", ios_base::in | ios_base::binary);
		train_labels.reserve(60000);
		for (size_t i = 0; i < 60000; i++) {
			file.read(tem, 1);
			train_labels.emplace_back(tem[0]);
		}
		file.close();

		file.open("DataSet\\MNIST\\t10k-images.idx3-ubyte", ios_base::in | ios_base::binary);
		test_images.reserve(10000 * 28 * 28);
		for (size_t i = 0; i < 10000; i++) {
			file.read(tem, 28 * 28);
			for (size_t j = 0; j < 28 * 28; j++) test_images.emplace_back((unsigned char)tem[j] / 255.f);
		}
		file.close();

		file.open("DataSet\\MNIST\\t10k-labels.idx1-ubyte", ios_base::in | ios_base::binary);
		test_labels.reserve(10000);
		for (size_t i = 0; i < 10000; i++) {
			file.read(tem, 1);
			test_labels.emplace_back(tem[0]);
		}
		file.close();

		cout << "終了" << endl;
	}//*/

	/*{
		fstream file;
		char tem[3 * 32 * 32];
		cout << "読み込み中...   ";

		train_images.reserve(50000 * 3 * 32 * 32);
		train_labels.reserve(50000);
		for (size_t i = 0; i < 50000; i++) {
			if (i % 10000 == 0) {
				file.close();
				file.open("DataSet\\CIFAR-10\\data_batch_" + to_string(i / 10000 + 1) + ".bin", ios_base::in | ios_base::binary);
			}
			file.read(tem, 1);
			train_labels.emplace_back(tem[0]);
			file.read(tem, 3 * 32 * 32);
			for (size_t j = 0; j < 3 * 32 * 32; j++) train_images.emplace_back((unsigned char)tem[j] / 255.f);
		}
		file.close();

		test_images.reserve(10000 * 3 * 32 * 32);
		test_labels.reserve(10000);
		file.open("DataSet\\CIFAR-10\\test_batch.bin", ios_base::in | ios_base::binary);
		for (size_t i = 0; i < 10000; i++) {
			file.read(tem, 1);
			test_labels.emplace_back(tem[0]);
			file.read(tem, 3 * 32 * 32);
			for (size_t j = 0; j < 3 * 32 * 32; j++) test_images.emplace_back((unsigned char)tem[j] / 255.f);
		}
		file.close();

		cout << "終了" << endl;
	}//*/

	
	const int N = 32;

	NeuralNetwork NN({28, 28, 1}, N);
	NN.AddLayer(new DenseLayer      (128)     , {1, 1, N}, {128, 1, 1});
	NN.AddLayer(new BiasLayer       ()        , {1, 1, N}, {128, 1, 1});
	NN.AddLayer(new ReluLayer       ()        , {1, 1, N}, {128, 1, 1});
	NN.AddLayer(new DropoutLayer    (0.2f)    , {1, 1, N}, {128, 1, 1});
	NN.AddLayer(new DenseLayer      (10)      , {1, 1, N}, {32, 1, 1});
	NN.AddLayer(new BiasLayer       ()        , {1, 1, N}, {32, 1, 1});
	NN.AddLayer(new SoftmaxLayer    ()        , {1, 1, N}, {32, 1, 1});//*/
	
	/*NeuralNetwork NN({32, 32, 3}, N);
	NN.AddLayer(new ConvolutionLayer(1, 1, 32), {8, 8, N}, {4, 4, 32});
	NN.AddLayer(new ReluLayer       ()        , {8, 8, N}, {4, 4, 32});
	NN.AddLayer(new MPoolingLayer   (2, 2)    , {8, 8, N}, {2, 2, 32});
	NN.AddLayer(new ConvolutionLayer(1, 1, 64), {8, 8, N}, {2, 2, 64});
	NN.AddLayer(new ReluLayer       ()        , {8, 8, N}, {2, 2, 64});
	NN.AddLayer(new MPoolingLayer   (2, 2)    , {4, 4, N}, {2, 2, 64});
	NN.AddLayer(new ConvolutionLayer(1, 1, 64), {4, 4, N}, {2, 2, 64});
	NN.AddLayer(new ReluLayer       ()        , {4, 4, N}, {2, 2, 64});
	NN.AddLayer(new DenseLayer      (64)      , {1, 1, N}, {64, 1, 1});
	NN.AddLayer(new BiasLayer       ()        , {1, 1, N}, {64, 1, 1});
	NN.AddLayer(new ReluLayer       ()        , {1, 1, N}, {64, 1, 1});
	NN.AddLayer(new DenseLayer      (10)      , {1, 1, N}, {32, 1, 1});
	NN.AddLayer(new BiasLayer       ()        , {1, 1, N}, {32, 1, 1});
	NN.AddLayer(new SoftmaxLayer    ()        , {1, 1, N}, {32, 1, 1});//*/

	NN.DeviceMalloc();

	//NN.Import("MyNN.bin");
	NN.NNReset();

	NN.fit(train_images, train_labels, 15);
	NN.Export("MyNN.bin");

	NN.test(test_images, test_labels);

	return 0;
}

