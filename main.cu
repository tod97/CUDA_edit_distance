#include <iostream>
#include <chrono>
using namespace std;
using namespace chrono;

// MACROS
#define MIN(x, y) (x < y ? x : y)
#define MAX(x, y) (x > y ? x : y)

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
	int n = 10;
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