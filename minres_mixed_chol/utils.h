/*
    To set the precision of the matrix-vector products to fp32/fp64,
    you must make the following changes to the four statements that follow:
        1) typedef float/double fp_matvec; 
        2) #define COMPUTE_MATVEC CUDA_R_32F/CUDA_R_64F
        3) #define ZERO_MATVEC 0.0f/0.0
        4) #define ONE_MATVEC  1.0f/1.0
*/
typedef double fp_matvec; 
#define COMPUTE_MATVEC CUDA_R_64F
#define ZERO_MATVEC 0.0
#define ONE_MATVEC  1.0

/*
    To set the precision of the preconditioner solves to fp32/fp64,
    you must make the following changes to the four statements that follow:
        1) typedef float/double fp_solve; 
        2) #define COMPUTE_SOLVE CUDA_R_32F/CUDA_R_64F
        4) #define ONE_MATVEC  1.0f/1.0
*/
typedef float fp_solve; 
#define COMPUTE_SOLVE CUDA_R_32F
#define ONE_SOLVE   1.0f

/*//////////////////////
    STRUCT DEFINITIONS
*///////////////////////

/*
    Represents a saddle-point system of the form:
        [A B'; B 0] x = b
    preconditioned by [M1 0; 0 M2].
    
    A is stored as an n x n sparse matrix in CSC format. 
    B is stored as an m x n sprase matrix in CSC format.
    R1 = chol(M1) is stored as an n x n sparse matrix in COO format.
    R2 = chol(M2) is stored as an m x m sparse matrix in COO format.
    
    The specified sparsity formats were chosen based on compatibility with the
    CUDA and CUSPARSE functions.
*/
struct SaddlePointSystem {
    mat_sparse_t *A;  // in CSC format
    mat_sparse_t *B;  // in CSC format
    mat_sparse_t *R1; // chol factor for M1 in COO format
    mat_sparse_t *R2; // chol factor for M2 in COO format
    double       *b;
    int           n;
    int           m;
};
/*
    Contains the params required to perform a matvec prod using the CUDA library functions
*/
struct CudaMatVecProdParams {
    cusparseHandle_t     handle;
    cusparseSpMatDescr_t mat;
    cusparseDnVecDescr_t vec1;
    cusparseDnVecDescr_t vec2;  
    fp_matvec            alpha;
    fp_matvec            beta; 
    size_t               bufferSize;
    size_t               bufferSizeT;
    int                 *dMat_rows;
    int                 *dMat_cols;
    fp_matvec           *dMat_vals;
    fp_matvec           *dVec1; // n by 1
    fp_matvec           *dVec2; // m by 1
    void                *dBuffer;
    void                *dBufferT;
};
/*
    Contains the params required to perform a preconditioenr solve using the CUDA library functions
*/
struct CudaPrecondSolveParam {
    cusparseHandle_t     handle;
    cusparseSpMatDescr_t matR;
    cusparseDnVecDescr_t vec1;
    cusparseDnVecDescr_t vec2;
    cusparseSpSVDescr_t  temp;
    cusparseSpSVDescr_t  tempT;
    fp_solve             alpha;
    size_t               bufferSize;
    size_t               bufferSizeT;
    int                 *dR_rows;
    int                 *dR_cols;
    fp_solve            *dR_vals;
    fp_solve            *dVec1;
    fp_solve            *dVec2;
    void                *dBuffer;
    void                *dBufferT;
};

/*
    Checks that the input CUDA function behaves accordingly.
*/
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

/*
    Checks that the input CUSPARSE function behaves accordingly.
*/
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

/*/////////////////////////
    FUNCTION DECLARATIONS
*//////////////////////////

/*
    MATIO-RELATED DECLARATIONS
*/
// returns an int value corresponding to an fp64 scalar under name from the provided matfile
int     get_scalar_from_matfile(mat_t *matfile, char *name);
// returns an fp64 vector of size m corresponding to an fp64 vector of size n under name from the provided matfile
double* get_vector_from_matfile(mat_t *matfile, char *name, int n, int m);
// frees all the allocated space for the provided system and prints success message is print is set to 1 
void    free_saddle_point_system(struct SaddlePointSystem *system, int print);
// opens a matfile with the name file, allocates the required space for its components under system, and prints success message is print is set to 1 
void    initialize_saddle_point_system(struct SaddlePointSystem *system, char *file, int print);

/*
    LINEAR ALGEBRA DECLARATIONS
*/
// returns an fp64 value corresponding to the dot product of the fp64 vectors v1 and v2, each of length n
double  dot_prod(double *v1, double *v2, int n);
// returns an fp64 value corresponding to the 2-norm of the fp64 vector v of length n
double  norm2(double *v, int n);
// returns an fp64 value corresponding to the 2-norm of the fp64 vector [s1, s2]
double  norm2_1d(double s1, double s2);
// computes b = a*x + b where x and b are both fp64 vectors of length n and a is an fp64 scalar
void    axpy(double a, double *x, double *b, int n);
// computes v_out = a*v_in where v_in and v_out are fp64 vectors of length n and a is an fp64 scalar
void    scalar_prod(double a, double *v_in, double *v_out, int n);

/*
    CUDA-RELATED DECLARATIONS
*/
// performs a test run of saddle_matvec_prod to ensure that all CUDA library functions operate properly
void    check_saddle_matvec_prod(struct CudaMatVecProdParams *paramsA, struct CudaMatVecProdParams *paramsB, 
                                 struct SaddlePointSystem *system, double *v, int print); 
// performs a test run of precond_solve to ensure that all CUDA library functions operate properly
void    check_saddle_precond_solve(struct CudaPrecondSolveParam *paramsM1, struct CudaPrecondSolveParam *paramsM2,
                                   struct SaddlePointSystem *system, double *v, int print); 
// frees all the allocated space for the provided matvec params
int     free_matvec_params(struct CudaMatVecProdParams *params, char *name, int includeTranpose, int print);
// frees all the allocated space for the provided solver params
int     free_solver_params(struct CudaPrecondSolveParam *params, char *name, int print);
// allocates the required space for the matvec params
int     initialize_matvec_params(struct CudaMatVecProdParams *params, mat_sparse_t *mat, int nrows, int ncols,
                                 char *name, int includeTranpose, int print);
// allocates the required space for the solver params
int     initialize_solver_params(struct CudaPrecondSolveParam *params, mat_sparse_t *matR, int n, char *name, int print);
// performs a matvec prod for the saddle point system corresponding to the provided params
//      1) compute v(1:n) = A*v(1:n) + B'*v(n+1:end) 
//      2) compute v(n+1:m) = B*v(1:n)
void    saddle_matvec_prod(struct CudaMatVecProdParams *paramsA, struct CudaMatVecProdParams *paramsB, 
                           struct SaddlePointSystem *system, double *v_in, double* v_out, int print);
// performs a preconditioner solve for the saddle point system corresponding to the provided params
//      1) compute M1\v1 = R1\(R1'\v1) where v1 = v(1:n) and M1 is n x n
//      2) compute M2\v2 = R2\(R2'\v2) where v2 = v(n+1:end)
void    saddle_precond_solve(struct CudaPrecondSolveParam *paramsM1, struct CudaPrecondSolveParam *paramsM2, 
                             struct SaddlePointSystem *system, double *v_in, double *v_out, int print);