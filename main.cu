#include <iostream>
#include <chrono>
using namespace std;
using namespace chrono;

// MACROS
#define MIN(x, y) (x < y ? x : y)

string generateWord(int n)
{
	string word = "";
	for (int i = 0; i < n; i++)
	{
		word += (char)('a' + rand() % 26);
	}
	return word;
}

int sequentialDistance(string A, string B)
{
	unsigned int lenA = A.size();
	unsigned int lenB = B.size();

	unsigned int **D = new unsigned int *[lenA + 1];
	for (int i = 0; i < lenA + 1; i++)
		D[i] = new unsigned int[lenB + 1];

	for (int i = 0; i < lenA + 1; i++)
		D[i][0] = i;
	for (int j = 1; j < lenB + 1; j++)
		D[0][j] = j;

	for (int i = 1; i < lenA + 1; i++)
	{
		for (int j = 1; j < lenB + 1; j++)
		{
			if (A[i - 1] == B[j - 1])
			{
				D[i][j] = D[i - 1][j - 1];
			}
			else
			{
				D[i][j] = 1 + min(min(D[i - 1][j], D[i][j - 1]), D[i - 1][j - 1]);
			}
		}
	}

	return D[lenA][lenB];
}

__global__ void editDistKernel(char *devA, char *devB, int lenA, int lenB, unsigned int *devPPrevDiag, unsigned int *devPrevDiag, unsigned int *devCurrDiag, int diagIdx)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int diagSize = (diagIdx <= lenA + 1) ? diagIdx : ((2 * (lenA+1)) - diagIdx);

	/* if (tid == 0) {
		printf("diagSize: %d\n", diagSize);
		for (int j = 0; j <= lenA; j++) printf("%d ", devPPrevDiag[j]);
		printf("-----------\n");
		for (int j = 0; j <= lenA; j++) printf("%d ", devPrevDiag[j]);
		printf("\n");
	} */

	if (tid < diagSize) {
		if (diagIdx <= lenA + 1) {
			if (tid == 0)
				devCurrDiag[tid] = devPrevDiag[tid] + 1;
			if (tid == diagSize - 1)
				devCurrDiag[tid] = devPrevDiag[tid - 1] + 1;

			if (tid > 0 && tid < diagSize - 1) {
				if (devA[tid - 1] != devB[diagSize - tid - 2])
					devCurrDiag[tid] = 1 + MIN(devPrevDiag[tid - 1], MIN(devPPrevDiag[tid - 1], devPrevDiag[tid]));
				else
					devCurrDiag[tid] = devPPrevDiag[tid - 1];
			}
		} else {
			int pprevIdx = (lenA - diagSize == 0) ? tid : tid + 1;
			if (devA[tid + lenA - diagSize] != devB[lenA - tid - 1]) {
				devCurrDiag[tid] = 1 + MIN(devPrevDiag[tid], MIN(devPPrevDiag[pprevIdx], devPrevDiag[tid + 1]));
			}
			else {
				devCurrDiag[tid] = devPPrevDiag[pprevIdx];
			}
		}
	}
}

int parallelDistance(const char *A, const char *B, int lenA, int lenB)
{
	int distance;
	char *devA;
	char *devB;
	unsigned int *devCurrDiag;
	unsigned int *devPrevDiag;
	unsigned int *devPPrevDiag;

	unsigned int *currDiag = new unsigned int[lenA + 1];
	unsigned int *prevDiag = new unsigned int[lenA + 1];
	unsigned int *pprevDiag = new unsigned int[lenA + 1];
	// Init first two diagonals
	pprevDiag[0] = 0;
	prevDiag[0] = 1;
	prevDiag[1] = 1;

	// CUDA Alloc
	cudaMalloc((void **)&devA, (lenA + 1) * sizeof(char));
	cudaMalloc((void **)&devB, (lenB + 1) * sizeof(char));
	cudaMalloc((void **)&devCurrDiag, (lenA + 1) * sizeof(unsigned int));
	cudaMalloc((void **)&devPrevDiag, (lenA + 1) * sizeof(unsigned int));
	cudaMalloc((void **)&devPPrevDiag, (lenA + 1) * sizeof(unsigned int));

	// CUDA copy into device
	cudaMemcpy((void *)devA, (void *)A, (lenA + 1) * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)devB, (void *)B, (lenB + 1) * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)devPPrevDiag, (void *)pprevDiag, (lenA + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)devPrevDiag, (void *)prevDiag, (lenA + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);

	for (int i = 3; i < 2 * (lenA + 1); i++)
	{
		int blockSize = min(i, 1024);
		int gridSize = ceil(float(i) / blockSize);

		editDistKernel<<<gridSize, blockSize>>>(devA, devB, lenA, lenB, devPPrevDiag, devPrevDiag, devCurrDiag, i);

		unsigned int *tmp = devPPrevDiag;
		devPPrevDiag = devPrevDiag;
		devPrevDiag = devCurrDiag;
		devCurrDiag = tmp;
	}
	// CUDA get result from device
	cudaMemcpy((void *)&distance, (void *)&devPrevDiag[0], 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// CUDA free device memory
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devPPrevDiag);
	cudaFree(devPrevDiag);
	cudaFree(devCurrDiag);

	return distance;
}

int main()
{
	int n = 10000;
	string A = generateWord(n);
	string B = generateWord(n);

	cout << "--------- STRING LENGTH = " << n << " ---------" << endl;
	// SEQUENTIAL
	auto start = system_clock::now();
	int distance = sequentialDistance(A, B);
	auto end = system_clock::now();
	auto elapsed = duration_cast<milliseconds>(end - start);
	cout << "Sequential [d=" << distance << "]: " << elapsed.count() << "ms" << endl;

	// PARALLEL
	start = system_clock::now();
	distance = parallelDistance(A.c_str(), B.c_str(), n, n);
	end = system_clock::now();
	elapsed = duration_cast<milliseconds>(end - start);
	cout << "Parallel [d=" << distance << "]: " << elapsed.count() << "ms" << endl;
}