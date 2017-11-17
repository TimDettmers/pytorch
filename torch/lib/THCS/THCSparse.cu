#include "THCSparse.h"

void THCudaSparse_Xcoo2csr(THCState *state, const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr) {
  THAssertMsg((m <= INT_MAX) && (nnz <= INT_MAX),
    "cusparseXcoo2csr only supports m, nnz with the bound [val] <= %d",
    INT_MAX);
  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  THCusparseCheck(cusparseXcoo2csr(handle, coorowind, nnz, m, csrrowptr,
    TH_INDEX_BASE ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO
  ));
}

cusparseOperation_t convertTransToCusparseOperation(char trans) {
  if (trans == 't') return CUSPARSE_OPERATION_TRANSPOSE;
  else if (trans == 'n') return CUSPARSE_OPERATION_NON_TRANSPOSE;
  else if (trans == 'c') return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else {
    THError("trans must be one of: t, n, c");
    return CUSPARSE_OPERATION_TRANSPOSE;
  }
}

void adjustLd(char transb, int64_t m, int64_t n, int64_t k, int64_t *ldb, int64_t *ldc)
{
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    *ldc = m;

  if(transb_)
  {
    if(k == 1)
      *ldb = n;
  }
  else
  {
    if(n == 1)
      *ldb = k;
  }
}

/* Level 3 */
void THCudaSparse_Scsrmm2(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseScsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
#if TH_INDEX_BASE == 1
  cusparseSetMatIndexBase(&desc, CUSPARSE_INDEX_BASE_ONE);
#endif
  THCusparseCheck(cusparseScsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc));
}

void THCudaSparse_Dcsrmm2(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseDcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
#if TH_INDEX_BASE == 1
  cusparseSetMatIndexBase(&desc, CUSPARSE_INDEX_BASE_ONE);
#endif
  THCusparseCheck(cusparseDcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc));
}

void THCudaSparse_Scsrgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnza, float *csrvala, int *csrrowptra, int *csrcolinda, int64_t nnzb, float *csrvalb, int *csrrowptrb, int *csrcolindb, int64_t *nnzc, float *csrvalc, int *csrrowptrc, int *csrcolindc)
{

  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnza <= INT_MAX) && (nnzb <= INT_MAX),
    "cusparseScsrgemm only supports m, n, k, nnzA, nnzB, and nnzC with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnza = (int)nnza;
  int i_nnzb = (int)nnzb;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseMatDescr_t descrB;
  cusparseCreateMatDescr(&descrB);
  cusparseMatDescr_t descrC;
  cusparseCreateMatDescr(&descrC);

#if TH_INDEX_BASE == 1
  cusparseSetMatIndexBase(&descrA, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatIndexBase(&descrB, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatIndexBase(&descrC, CUSPARSE_INDEX_BASE_ONE);
#endif

	int baseC;
	int *nnzTotalDevHostPtr = 0;

	THCudaCheck(THCudaMalloc(state, (void**)(&csrrowptrc), m+1));

  THCusparseCheck(cusparseXcsrgemmNnz(handle, opa, opb, i_m, i_n, i_k,
          descrA, i_nnza, csrrowptra, csrcolinda,
          descrB, i_nnzb, csrrowptrb, csrcolindb,
          descrC, csrrowptrc, nnzTotalDevHostPtr));

  *nnzc = (int64_t)(*nnzTotalDevHostPtr);

	THCudaCheck(THCudaMalloc(state, (void**)(&csrcolindc), *nnzc));
	THCudaCheck(THCudaMalloc(state, (void**)(&csrvalc), *nnzc));

  THCusparseCheck(cusparseScsrgemm(handle, opa, opb, i_m, i_n, i_k,
        descrA, i_nnza, csrvala, csrrowptra, csrcolinda,
        descrB, i_nnzb, csrvalb, csrrowptrb, csrcolindb,
        descrC, csrvalc, csrrowptrc, csrcolindc));
}



void THCudaSparse_Dcsrgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnza, double *csrvala, int *csrrowptra, int *csrcolinda, int64_t nnzb, double *csrvalb, int *csrrowptrb, int *csrcolindb, int64_t *nnzc, double *csrvalc, int *csrrowptrc, int *csrcolindc)
{

  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnza <= INT_MAX) && (nnzb <= INT_MAX),
    "cusparseScsrgemm only supports m, n, k, nnzA, nnzB, and nnzC with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnza = (int)nnza;
  int i_nnzb = (int)nnzb;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseMatDescr_t descrB;
  cusparseCreateMatDescr(&descrB);
  cusparseMatDescr_t descrC;
  cusparseCreateMatDescr(&descrC);

#if TH_INDEX_BASE == 1
  cusparseSetMatIndexBase(&descrA, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatIndexBase(&descrB, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatIndexBase(&descrC, CUSPARSE_INDEX_BASE_ONE);
#endif

	int baseC, nnzC = 0; // nnzTotalDevHostPtr points to host memory int *nnzTotalDevHostPtr = &nnzC;
	int *nnzTotalDevHostPtr = &nnzC;

	printf("m: %d\n", m);
	printf("nnza: %ld\n", i_nnza);
	printf("nnzb: %ld\n", i_nnzb);

	THCudaCheck(THCudaMalloc(state, (void**)(&csrrowptrc), m+1));

	printf("post pointer \n");

  printf("m: %d \n", i_m);
  printf("n: %d \n", i_n);
  printf("k: %d \n", i_k);
  printf("nnza: %d \n", i_nnza);
  printf("nnzb: %d \n", i_nnzb);

  THCusparseCheck(cusparseXcsrgemmNnz(handle, opa, opb, i_m, i_n, i_k,
          descrA, i_nnza, csrrowptra, csrcolinda,
          descrB, i_nnzb, csrrowptrb, csrcolindb,
          descrC, csrrowptrc, nnzTotalDevHostPtr));

  if (NULL != nnzTotalDevHostPtr)
  {
			printf("post nnzc %d \n", nnzC);
      nnzC = *nnzTotalDevHostPtr;
			printf("assigned!\n");
			printf("post nnzc %d \n", nnzC);
	}
	else
	{

		printf("post nnzc %d \n", *nnzTotalDevHostPtr);
		printf("post nnzc %d \n", nnzC);
	}


  *nnzc = int64_t(*nnzTotalDevHostPtr);

  printf("nnzc2: %ld", nnzc);

	THCudaCheck(THCudaMalloc(state, (void**)(&csrcolindc), nnzC));
	THCudaCheck(THCudaMalloc(state, (void**)(&csrvalc), nnzC));

	printf("post alloc \n");

  THCusparseCheck(cusparseDcsrgemm(handle, opa, opb, i_m, i_n, i_k,
        descrA, i_nnza, csrvala, csrrowptra, csrcolinda,
        descrB, i_nnzb, csrvalb, csrrowptrb, csrcolindb,
        descrC, csrvalc, csrrowptrc, csrcolindc));

	printf("post gemm \n");
}

/* format conversion */
void THCudaSparse_CreateIdentityPermutation(THCState *state, int64_t nnz, int *P) {
  THAssertMsg((nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <= %d",
    INT_MAX);
  int i_nnz = (int)nnz;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  cusparseCreateIdentityPermutation(handle, i_nnz, P);
}

void THCudaSparse_Xcsrsort_bufferSizeExt(THCState *state, int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes)
{
  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  THCusparseCheck(cusparseXcsrsort_bufferSizeExt(handle, i_m, i_n, i_nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));
}

void THCudaSparse_Xcsrsort(THCState *state, int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer)
{
  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort only supports m, n, nnz with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
#if TH_INDEX_BASE == 1
  cusparseSetMatIndexBase(&desc, CUSPARSE_INDEX_BASE_ONE);
#endif
  THCusparseCheck(cusparseXcsrsort(handle, i_m, i_n, i_nnz, desc, csrRowPtr, csrColInd, P, pBuffer));
}

void THCudaSparse_Xcoosort_bufferSizeExt(THCState *state, int64_t m, int64_t n, int64_t nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes)
{
  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcoosort_bufferSizeExt only supports m, n, nnz with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  THCusparseCheck(cusparseXcoosort_bufferSizeExt(handle, i_m, i_n, i_nnz, cooRows, cooCols, pBufferSizeInBytes));
}

void THCudaSparse_XcoosortByRow(THCState *state, int64_t m, int64_t n, int64_t nnz, int *cooRows, int *cooCols, int *P, void *pBuffer)
{
  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "XcoosortByRow only supports m, n, nnz with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  THCusparseCheck(cusparseXcoosortByRow(handle, i_m, i_n, i_nnz, cooRows, cooCols, P, pBuffer));
}
