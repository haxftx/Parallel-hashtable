#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include <math.h>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"


using namespace std;
/**
 * Funtion hash
 */
__device__ unsigned int fHash(int key) {
	return (unsigned long long)key *  110477914016779llu % 452517535812813007llu;
}
/**
 *Function constructor GpuHashTable
 */
GpuHashTable::GpuHashTable(int size) {
	glbGpuAllocator->_cudaMallocManaged((void **) &GPUHTable.data, size * sizeof(pair<int, int>));
	cudaMemset(GPUHTable.data, 0, size * sizeof(pair<int, int>));
	GPUHTable.count = 0;
	GPUHTable.size = size;	
}
/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree(GPUHTable.data);
	GPUHTable.count = GPUHTable.size = 0;	
}
/**
 * Kernel funtion reshape
 */
__global__ void kernel_reshape(pair<int, int> *oldData, pair<int, int> *newData, int oldSize, int newSize) {
	int  i = 0, idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < oldSize && oldData[idx].first != 0) { // replace the old key:value to new position
		while (i < oldSize) { // search liniar free slot
			if (atomicCAS(&newData[(fHash(oldData[idx].first) + i) % newSize].first, 0, oldData[idx].first) == 0) {
				atomicExch(&newData[(fHash(oldData[idx].first) + i) % newSize].second, oldData[idx].second);
				return;
			}
			i++;
		}
	}
}
/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	pair<int, int> *data; // allocation of memory
	glbGpuAllocator->_cudaMallocManaged((void **) &data, numBucketsReshape * sizeof(pair<int, int>));
	cudaMemset(data, 0, numBucketsReshape * sizeof(pair<int, int>));
	// call kernel
	unsigned int blocks = ceil(1.0 * numBucketsReshape / THREADS_PER_BLOCK);
	kernel_reshape<<<blocks, THREADS_PER_BLOCK>>>(GPUHTable.data, data, GPUHTable.size, numBucketsReshape);
	cudaDeviceSynchronize(); // wait
	cudaCheckError(); // ceck error
	GPUHTable.size = numBucketsReshape; // update size
	glbGpuAllocator->_cudaFree(GPUHTable.data);
	GPUHTable.data = data; // update data
}
/**
 * Kernel funtion insertBatch
 */
__global__ void kernel_insert(int *keys, int *values, int numKeys, Htable GPUHTable) {
	int key, i = 0, idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numKeys) { // insert key:value
		while (i < GPUHTable.size) { // search liniar free slot
			key = atomicCAS(&(GPUHTable.data[(fHash(keys[idx]) + i) % GPUHTable.size].first), 0, keys[idx]);
			if (key == 0) { // free slot
				atomicExch(&(GPUHTable.data[(fHash(keys[idx]) + i) % GPUHTable.size].second), values[idx]);
				return;
			} else if (key == keys[idx]) { // update
				atomicExch(&(GPUHTable.data[(fHash(keys[idx]) + i) % GPUHTable.size].second), values[idx]);
				GPUHTable.count--;
				return;
			}
			i++;
		}
	}
}
/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *deviceKeys, *deviceValues; // allocation of memory
	glbGpuAllocator->_cudaMalloc((void **)&deviceKeys, numKeys * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void **)&deviceValues, numKeys * sizeof(int));

	if (!deviceKeys || !deviceValues) { // ceck memory
		return false;
	}
	if ((GPUHTable.count + numKeys) >= GPUHTable.size) // ceck load factor
		reshape((GPUHTable.count + numKeys) / MIN_LOAD);

	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	// call kernel
	unsigned int blocks = ceil(1.0 * numKeys / THREADS_PER_BLOCK);
	kernel_insert<<<blocks, THREADS_PER_BLOCK>>>(deviceKeys, deviceValues, numKeys, GPUHTable);
	cudaDeviceSynchronize(); // wait
	cudaCheckError(); // ceck error
	GPUHTable.count += numKeys; // update count
	glbGpuAllocator->_cudaFree(deviceKeys);// free of memory
	glbGpuAllocator->_cudaFree(deviceValues);
	return true;
}
/**
 * Kernel function getBatch
 */
__global__ void kernel_get(int *keys, int *values, int numKeys, Htable GPUHTable) {
	int i = 0, idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numKeys) { // find value for key
		while(i < GPUHTable.size) { // find liniar key
			if (GPUHTable.data[(fHash(keys[idx]) + i) % GPUHTable.size].first == keys[idx]) {
				atomicExch(&values[idx], GPUHTable.data[(fHash(keys[idx]) + i) % GPUHTable.size].second);
				return;
			}
			i++;
		}
		values[idx] = -1;
	}
}
/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys, *deviceValues, *hostValues; // alocation of memory
	glbGpuAllocator->_cudaMalloc((void **)&deviceKeys, numKeys * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void **)&deviceValues, numKeys * sizeof(int));
	hostValues = (int *)malloc(numKeys * sizeof(int));
	if (!deviceKeys || !deviceValues || !hostValues) // ceck memory
		return NULL;
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	// call kernel
	unsigned int blocks = ceil(1.0 * numKeys / THREADS_PER_BLOCK);
	kernel_get<<<blocks, THREADS_PER_BLOCK>>>(deviceKeys, deviceValues, numKeys, GPUHTable);
	cudaDeviceSynchronize(); // wait
	cudaCheckError(); // ceck error
	cudaMemcpy(hostValues, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	glbGpuAllocator->_cudaFree(deviceValues); // free of memory
	glbGpuAllocator->_cudaFree(deviceKeys);
	return hostValues;
}
/**
 * Funtion get load factor
 */
float GpuHashTable::load_factor(void) { // calculate load factor
	if (GPUHTable.size == 0)
		return 0;
	return	GPUHTable.count * 1.0 / GPUHTable.size;
}

