#ifndef _HASHCPU_
#define _HASHCPU_

#include <vector>
#include <utility>

#define THREADS_PER_BLOCK 512
#define MIN_LOAD 0.5f
#define MAX_LOAD 0.99f
using namespace std;

#define cudaCheckError() { \
	cudaError_t e=cudaGetLastError(); \
	if(e!=cudaSuccess) { \
		cout << "Cuda failure " << __FILE__ << ", " << __LINE__ << ", " << cudaGetErrorString(e); \
		exit(0); \
	 }\
}


typedef struct Htable {
	int count, size;
	pair<int, int> *data;
} Htable;

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		float load_factor(void);
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		~GpuHashTable();
		Htable GPUHTable;
};

#endif

