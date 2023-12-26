#include <iostream>
#include <chrono>
using namespace std;
using namespace chrono;

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

int main()
{
   string A = generateWord(10000);
   string B = generateWord(10000);

   // SEQUENTIAL
   auto start = system_clock::now();
   int distance = sequentialDistance(A, B);
   auto end = system_clock::now();
   auto elapsed = duration_cast<milliseconds>(end - start);
   cout << "seq Distance (n=" << A.size() << ", d=" << distance << "): " << elapsed.count() << "ms" << endl;
}