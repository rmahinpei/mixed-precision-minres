#include <cuda_runtime_api.h>
#include <cusparse.h>    
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <matio.h>
#include <math.h>
#include "utils.h"

/*
    MATIO-RELATED IMPLEMENTATIONS
*/
int get_scalar_from_matfile(mat_t *matfile, char *name) {
    matvar_t *var = Mat_VarRead(matfile, name);
    double *var_data = var->data;
    return (int)(*var_data);
}

double* get_vector_from_matfile(mat_t *matfile, char *name, int n, int m) {
    double *f = Mat_VarRead(matfile, (char *)"f")->data;
    double *b = calloc(n+m, sizeof(double));
    for (int i = 0; i < n; i++) {
        b[i] = f[i];
    }
    matvar_t *g_var = Mat_VarRead(matfile, (char *)"g");
    if (g_var != NULL) { 
        double *g = (double *)g_var->data;
        for (int i = 0; i < m; i++) {
            b[n + i] = g[i];
        }
    }
    free(f); free(g_var);
    return b;
}

void csc_to_coo(mat_sparse_t *mat) {
    int entryIndex = 0;
    int *jc = malloc(mat->nzmax*sizeof(int));
    for (int j = 0; j < (mat->njc)-1; j++) {
        int colStart = mat->jc[j];
        int colEnd   = mat->jc[j+1];
        for (int i = colStart; i < colEnd; i++) {
            jc[entryIndex] = j;
            entryIndex++;
        }
    }
    free(mat->jc);
    mat->jc  = jc;
    mat->njc = mat->nzmax;
}

fp_matvec* double_to_fp_matvec(double *v, int n) {
    fp_matvec *w = (fp_matvec *)malloc(n*sizeof(fp_matvec));
    for (int i = 0; i < n; i++) {
        w[i] = (fp_matvec)v[i];
    }
    return w;
}

void fp_matvec_to_double(fp_matvec *w, double *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = (double)w[i];
    }
    free(w);
}

fp_solve* double_to_fp_solve(double *v, int n) {
    fp_solve *w = (fp_solve *)malloc(n*sizeof(fp_solve));
    for (int i = 0; i < n; i++) {
        w[i] = (fp_solve)v[i];
    }
    return w;
}

void fp_solve_to_double(fp_solve *w, double *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = (double)w[i];
    }
    free(w);
}

void initialize_saddle_point_system(struct SaddlePointSystem *system, char *file, int print) {
    mat_t *matfile = Mat_Open(file, MAT_ACC_RDONLY);
    system->A  = Mat_VarRead(matfile, (char *)"A")->data;  system->A->data  = double_to_fp_matvec(system->A->data, system->A->nzmax);  
    system->B  = Mat_VarRead(matfile, (char *)"B")->data;  system->B->data  = double_to_fp_matvec(system->B->data, system->B->nzmax);  
    system->R1 = Mat_VarRead(matfile, (char *)"R1")->data; system->R1->data = double_to_fp_solve(system->R1->data, system->R1->nzmax);  
    system->R2 = Mat_VarRead(matfile, (char *)"R2")->data; system->R2->data = double_to_fp_solve(system->R2->data, system->R2->nzmax);  
    csc_to_coo(system->R1); csc_to_coo(system->R2);
    system->n  = get_scalar_from_matfile(matfile, (char *)"size_n");
    system->m  = get_scalar_from_matfile(matfile, (char *)"size_m");
    system->b  = get_vector_from_matfile(matfile, (char *)"f", system->n, system->m);
    if (print) {
        printf("n = %d \t m = %d\n", system->n, system->m);
        printf("nnz in A: %d \t nnz in B: %d\n", (int)system->A->nzmax, (int)system->B->nzmax);
        printf("nir in A: %d \t nir in B: %d\n", (int)system->A->nir, (int)system->B->nir);
        printf("njc in A: %d \t njc in B: %d\n", (int)system->A->njc, (int)system->B->njc);
        printf("Successfully initialized saddle-point system.\n");
    }
}

void free_saddle_point_system(struct SaddlePointSystem *system, int print) {
    free(system->A->ir);  free(system->A->jc);  free(system->A->data);
    free(system->B->ir);  free(system->B->jc);  free(system->B->data);
    free(system->R1->ir); free(system->R1->jc); free(system->R1->data);
    free(system->R2->ir); free(system->R2->jc); free(system->R2->data);
    free(system->A);      free(system->B);      free(system->b);
    free(system->R1);     free(system->R2);
    if (print) {
        printf("Successfully freed allocated memory for saddle-point system.\n");
    }
}

/*
    LINEAR ALGEBRA IMPLEMENTATIONS
*/
void scalar_prod(double a, double *v_in, double *v_out, int n) {
    for (int i = 0; i < n; i++) {
        v_out[i] = a*v_in[i];
    }
}

double dot_prod(double *v1, double *v2, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

void axpy(double a, double *x, double *b, int n) {
    for (int i = 0; i < n; i++) {
        b[i] = a*x[i] + b[i];
    }
}

double norm2(double *v, int n) {
    return sqrt(dot_prod(v, v, n));
}

double norm2_1d(double s1, double s2) {
    return sqrt(s1*s1 + s2*s2);
}


/*
    CUDA-RELATED IMPLEMENTATIONS
*/
int allocate_matvec_prod_memory(struct CudaMatVecProdParams *params, mat_sparse_t *mat, int nrows, int ncols,
                                  char *name, int print) {
    CHECK_CUDA( cudaMalloc((void**) &params->dMat_rows,  (mat->nir)  *sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &params->dMat_cols,  (mat->njc)  *sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &params->dMat_vals,  (mat->nzmax)*sizeof(fp_matvec)) )
    CHECK_CUDA( cudaMalloc((void**) &params->dVec1,      (ncols)     *sizeof(fp_matvec)) )
    CHECK_CUDA( cudaMalloc((void**) &params->dVec2,      (nrows)     *sizeof(fp_matvec)) )
    CHECK_CUDA( cudaMemcpy(params->dMat_rows, mat->ir,   (mat->nir)  *sizeof(int),       cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(params->dMat_cols, mat->jc,   (mat->njc)  *sizeof(int),       cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(params->dMat_vals, mat->data, (mat->nzmax)*sizeof(fp_matvec), cudaMemcpyHostToDevice) )
    if (print) {
        printf("Successfully allocated device memory for matvec prod with %s.\n", name);
    }
}


int initialize_matvec_params(struct CudaMatVecProdParams *params, mat_sparse_t *mat, int nrows, int ncols,
                             char *name, int includeTranpose, int print) {
    allocate_matvec_prod_memory(params, mat, nrows, ncols, name, print);
    params->alpha    = ONE_MATVEC;  
    params->beta     = ZERO_MATVEC;       
    params->handle   = NULL; 
    CHECK_CUSPARSE( cusparseCreate   (&params->handle) )
    CHECK_CUSPARSE( cusparseCreateCsc(&params->mat, nrows, ncols, mat->nzmax, 
                                      params->dMat_cols, params->dMat_rows, params->dMat_vals, 
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, COMPUTE_MATVEC) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&params->vec1, ncols, params->dVec1, COMPUTE_MATVEC) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&params->vec2, nrows, params->dVec2, COMPUTE_MATVEC) )
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                      params->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &params->alpha, params->mat, params->vec1, &params->beta, params->vec2, COMPUTE_MATVEC,
                                      CUSPARSE_SPMV_ALG_DEFAULT, &params->bufferSize) )
    CHECK_CUDA( cudaMalloc(&params->dBuffer, params->bufferSize) )
    if (includeTranpose) {
        CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                      params->handle, CUSPARSE_OPERATION_TRANSPOSE,
                                      &params->alpha, params->mat, params->vec2, &params->beta, params->vec1, COMPUTE_MATVEC,
                                      CUSPARSE_SPMV_ALG_DEFAULT, &params->bufferSizeT) )
        CHECK_CUDA( cudaMalloc(&params->dBufferT, params->bufferSizeT) )
    }
    if (print) {
        printf("Successfully initialized matvec params for %s.\n", name);
    }
}

int free_matvec_params(struct CudaMatVecProdParams *params, char *name, int includeTranpose, int print) {
    // destroy descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(params->mat) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(params->vec1) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(params->vec2) )
    CHECK_CUSPARSE( cusparseDestroy(params->handle) )
    // free allocated memory
    CHECK_CUDA( cudaFree(params->dBuffer) )
    CHECK_CUDA( cudaFree(params->dMat_rows) )
    CHECK_CUDA( cudaFree(params->dMat_cols) )
    CHECK_CUDA( cudaFree(params->dMat_vals) )
    CHECK_CUDA( cudaFree(params->dVec1) )
    CHECK_CUDA( cudaFree(params->dVec2) )
    if (includeTranpose) {
        CHECK_CUDA( cudaFree(params->dBufferT) )
    }
    if (print) {
        printf("Successfully freed allocated memory for matvec params of %s.\n", name);
    }
}

void check_saddle_matvec_prod(struct CudaMatVecProdParams *paramsA, struct CudaMatVecProdParams *paramsB, 
                              struct SaddlePointSystem *system, double *v, int print) {
    for (int i = 0; i < system->n+system->m; i++) {
        v[i] = 2;
    } 
    saddle_matvec_prod(paramsA, paramsB, system, v, v, 0);
    if (print) {
        printf("Successfully ran saddle-point system matrix-vector product.\n");
    }
}

void saddle_matvec_prod(struct CudaMatVecProdParams *paramsA, struct CudaMatVecProdParams *paramsB, 
                        struct SaddlePointSystem *system, double *v_in, double* v_out, int print) {
    fp_matvec *w = double_to_fp_matvec(v_in, system->n+system->m);
    // step 1: compute paramsA->dVec2 = A*v(1:n)  
    cudaMemcpy(paramsA->dVec1, w, system->n*sizeof(fp_matvec), cudaMemcpyHostToDevice);
    cusparseSpMV(paramsA->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &paramsA->alpha, paramsA->mat, paramsA->vec1, &paramsA->beta,
                 paramsA->vec2, COMPUTE_MATVEC, CUSPARSE_SPMV_ALG_DEFAULT, paramsA->dBuffer);
    // step 2: compute paramsA->dVec2 = A*v(1:n) + B'*v(n+1:end)  
    paramsB->beta = ONE_MATVEC;
    cudaMemcpy(paramsB->dVec2, w+system->n, system->m*sizeof(fp_matvec), cudaMemcpyHostToDevice);
    cusparseSpMV(paramsB->handle, CUSPARSE_OPERATION_TRANSPOSE,
                 &paramsB->alpha, paramsB->mat, paramsB->vec2, &paramsB->beta,
                 paramsA->vec2, COMPUTE_MATVEC, CUSPARSE_SPMV_ALG_DEFAULT, paramsB->dBufferT);
    // step 3: compute paramsB->dVec2 = v(n+1:m) = B*v(1:n)
    paramsB->beta = ZERO_MATVEC; 
    cusparseSpMV(paramsB->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &paramsB->alpha, paramsB->mat, paramsA->vec1, &paramsB->beta,
                 paramsB->vec2, COMPUTE_MATVEC, CUSPARSE_SPMV_ALG_DEFAULT, paramsB->dBuffer);
    // step 4: transfer results back to host memory
    //         v(1:n) = paramsA->dVec2 and v(n+1:end) = paramsB->dVec2   
    cudaMemcpy(w, paramsA->dVec2, system->n*sizeof(fp_matvec), cudaMemcpyDeviceToHost);
    cudaMemcpy(w+system->n, paramsB->dVec2, system->m*sizeof(fp_matvec), cudaMemcpyDeviceToHost);
    fp_matvec_to_double(w, v_out, system->n+system->m);
    if (print) {
        printf("Successfully computed saddle-point system matrix-vector product.\n");
    }
}


int allocate_solver_device_memory(struct CudaPrecondSolveParam *params, mat_sparse_t *matR, int n, char *name, int print) {
    CHECK_CUDA( cudaMalloc((void**) &params->dVec1,    (n)          *sizeof(fp_solve)) )
    CHECK_CUDA( cudaMalloc((void**) &params->dVec2,    (n)          *sizeof(fp_solve)) )
    CHECK_CUDA( cudaMalloc((void**) &params->dR_rows,  (matR->nir)  *sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &params->dR_cols,  (matR->njc)  *sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &params->dR_vals,  (matR->nzmax)*sizeof(fp_solve)) )
    CHECK_CUDA( cudaMemcpy(params->dR_rows, matR->ir,  (matR->nir)  *sizeof(int),      cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(params->dR_cols, matR->jc,  (matR->njc)  *sizeof(int),      cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(params->dR_vals, matR->data,(matR->nzmax)*sizeof(fp_solve), cudaMemcpyHostToDevice) )
    if (print) {
        printf("Successfully allocated device memory for precond solve with %s.\n", name);
    }
}

int free_solver_params(struct CudaPrecondSolveParam *params, char *name, int print) {
    // destroy descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(params->matR) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(params->vec1) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(params->vec2) )
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(params->temp));
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(params->tempT));
    CHECK_CUSPARSE( cusparseDestroy(params->handle) )

    // free allocated memory
    CHECK_CUDA( cudaFree(params->dBuffer) )
    CHECK_CUDA( cudaFree(params->dBufferT) )
    CHECK_CUDA( cudaFree(params->dR_rows) )
    CHECK_CUDA( cudaFree(params->dR_cols) )
    CHECK_CUDA( cudaFree(params->dR_vals) )
    CHECK_CUDA( cudaFree(params->dVec1) )
    CHECK_CUDA( cudaFree(params->dVec2) )
    if (print) {
        printf("Successfully freed allocated memory for precond solve with %s.\n", name);
    }
}

int initialize_solver_params(struct CudaPrecondSolveParam *params, mat_sparse_t *matR, int n, char *name, int print) {
    allocate_solver_device_memory(params, matR, n, name, print);
    params->alpha = ONE_SOLVE;
    CHECK_CUSPARSE( cusparseCreate(&params->handle) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&params->vec1, n, params->dVec1, COMPUTE_SOLVE) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&params->vec2, n, params->dVec2, COMPUTE_SOLVE) )
    // step 1: create Cholesky factor R (upper triangular)
    CHECK_CUSPARSE( cusparseCreateCoo(&params->matR, n, n, matR->nzmax, 
                                      params->dR_rows, params->dR_cols, params->dR_vals, 
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, COMPUTE_SOLVE) )
    cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_UPPER;
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(params->matR, CUSPARSE_SPMAT_FILL_MODE,
                                              &fillmode, sizeof(fillmode)) )
    cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(params->matR, CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diagtype, sizeof(diagtype)) )
    // step 2: for lower-triangular solve with R'
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&params->tempT) )
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                                params->handle, CUSPARSE_OPERATION_TRANSPOSE,
                                &params->alpha, params->matR, params->vec1, params->vec2, 
                                COMPUTE_SOLVE, CUSPARSE_SPSV_ALG_DEFAULT, params->tempT,
                                &params->bufferSizeT) )
    CHECK_CUDA( cudaMalloc(&params->dBufferT, params->bufferSizeT) )
    CHECK_CUSPARSE( cusparseSpSV_analysis(
                                params->handle, CUSPARSE_OPERATION_TRANSPOSE,
                                &params->alpha, params->matR, params->vec1, params->vec2, 
                                COMPUTE_SOLVE, CUSPARSE_SPSV_ALG_DEFAULT, params->tempT, params->dBufferT) )
    
    // step 4: for upper-triangular solve with R
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&params->temp) )
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                                params->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &params->alpha, params->matR, params->vec2, params->vec1, 
                                COMPUTE_SOLVE, CUSPARSE_SPSV_ALG_DEFAULT, params->temp,
                                &params->bufferSize ) )
    CHECK_CUDA( cudaMalloc(&params->dBuffer, params->bufferSize) )
    CHECK_CUSPARSE( cusparseSpSV_analysis(
                                params->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &params->alpha, params->matR, params->vec2, params->vec1, 
                                COMPUTE_SOLVE, CUSPARSE_SPSV_ALG_DEFAULT, params->temp, params->dBuffer) )
    if (print) {
        printf("Successfully initialized solver params for %s.\n", name);
    }
}

void precond_solve(struct CudaPrecondSolveParam *params, int n, fp_solve *w) {
    cudaMemcpy(params->dVec1, w, n*sizeof(fp_solve), cudaMemcpyHostToDevice);
    cusparseSpSV_solve(params->handle, CUSPARSE_OPERATION_TRANSPOSE,
                       &params->alpha, params->matR, params->vec1, params->vec2, COMPUTE_SOLVE,
                       CUSPARSE_SPSV_ALG_DEFAULT, params->tempT);
    cusparseSpSV_solve(params->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       &params->alpha, params->matR, params->vec2, params->vec1, COMPUTE_SOLVE,
                       CUSPARSE_SPSV_ALG_DEFAULT, params->temp);
    cudaMemcpy(w, params->dVec1, n*sizeof(fp_solve), cudaMemcpyDeviceToHost);
}

void saddle_precond_solve(struct CudaPrecondSolveParam *paramsM1, struct CudaPrecondSolveParam *paramsM2, 
                          struct SaddlePointSystem *system, double *v_in, double *v_out, int print) {
    fp_solve *w = double_to_fp_solve(v_in, system->n+system->m);
    // evaluate M1\v1 = R1\(R1'\v1) where v1 = v(1:n)
    precond_solve(paramsM1, system->n, w);
    // evaluate M2\v2 = R2\(R2'\v2) where v2 = v(n+1:end)
    precond_solve(paramsM2, system->m, w+system->n);
    fp_solve_to_double(w, v_out, system->n+system->m);
    if (print) {
        printf("Successfully computed saddle-point system preconditioner solve.\n");
    }
}

void check_saddle_precond_solve(struct CudaPrecondSolveParam *paramsM1, struct CudaPrecondSolveParam *paramsM2,
                                struct SaddlePointSystem *system, double *v, int print) {
    for (int i = 0; i < system->n+system->m; i++) {
        v[i] = 2;
    } 
    saddle_precond_solve(paramsM1, paramsM2, system, v, v, 0);
    if (print) {
        printf("Successfully ran saddle-point system preconditioner solve.\n");
    }
}