#include <cuda_runtime_api.h>
#include <sys/resource.h>
#include <cusparse.h>
#include <string.h>      
#include <stdlib.h>
#include <stdio.h>
#include <matio.h>
#include <math.h>
#include "utils.h"

/*
    MINRES implementation closely following Saunder's MINRES implementation (MATLAB version).
    Implementation has been adjusted to support
        1) saddle-point systems having C = 0 and
        2) used-specified precision for the matrix-vector products and preconditioner solves.
    To specify the precision of the matrix-vector products and preconditioner solves, 
    adjust the user-defined types and constants in utils.h accordingly.

    INPUTS:
        char *file: path to .mat file with respect to curr dir.
        int print: includes additional print statements if set to 1.
    OUTPUTS:
        N/A. 
        The total execution time and the relative residual norm at each iteration is printed to the CLI.
*/
void minres(char *file, int print) {
    struct SaddlePointSystem system;
    initialize_saddle_point_system(&system, file, print);

    struct CudaMatVecProdParams paramsA, paramsB;
    initialize_matvec_params(&paramsA, system.A, system.n, system.n, "A", 0, print);
    initialize_matvec_params(&paramsB, system.B, system.m, system.n, "B", 1, print);

    struct CudaPrecondSolveParam paramsM1, paramsM2;
    initialize_solver_params(&paramsM1, system.R1, system.n, "M1", print); 
    initialize_solver_params(&paramsM2, system.R2, system.m, "M2", print);

    // initialize hyper-parameters
    int itnlim      = 500;     double rtol     = 1e-6;     double rtol0    = rtol;         

    // initialize scalars needed for iterations
    double alpha    = 0.0;     double beta     = 0.0;      double gamma    = 0.0;
    double eps      = 0.0;     double delta    = 0.0;      double phi      = 0.0;
    double bnorm    = 0.0;     double rnorm    = 0.0;      double rnormk   = 0.0;
    double betaPrev = 0.0;     double epsPrev  = 0.0;      double s        = 0.0;
    double dbar     = 0.0;     double gbar     = 0.0;      double phibar   = 0.0; 
    double cs       =-1.0;     double sn       = 0.0;      double *temp    = NULL; // for temp calcs

    int N = system.n + system.m;  
    // initialize vectors needed for iterations
    double *x  = (double *)calloc(N,sizeof(double));
    double *y  = (double *)malloc(N*sizeof(double)); 
    double *r1 = (double *)malloc(N*sizeof(double)); memcpy(r1, system.b, N*sizeof(double));
    double *r2 = (double *)malloc(N*sizeof(double)); memcpy(r2, system.b, N*sizeof(double));
    double *v  = (double *)calloc(N,sizeof(double)); // temp vector
    double *w  = (double *)calloc(N,sizeof(double)); 
    double *w1 = (double *)calloc(N,sizeof(double)); // temp vector 
    double *w2 = (double *)calloc(N,sizeof(double));

    // check that CUDA API operates properly
    check_saddle_matvec_prod  (&paramsA,  &paramsB,  &system, v, print);
    check_saddle_precond_solve(&paramsM1, &paramsM2, &system, v, print);

    saddle_precond_solve(&paramsM1, &paramsM2, &system, system.b, y, 0);
    bnorm  = norm2(system.b, N); 
    rnorm  = bnorm; // assuming x0 = zeros(N,1)
    beta   = sqrt(dot_prod(system.b, y, N)); 
    phibar = beta; 

    printf("-----------------------------\n");
    printf("Starting MINRES iterations...\n");
    struct rusage rtime0, rtime1; 
    getrusage(RUSAGE_SELF, &rtime0); // record start time
    for (int itn = 1; itn <= itnlim; itn++) {
        s = 1/beta;
        scalar_prod(s, y, v, N); 
        saddle_matvec_prod(&paramsA, &paramsB, &system, v, y, 0);
        if (itn >= 2) axpy(-beta/betaPrev, r1, y, N); 

        alpha = dot_prod(v, y, N); 
        axpy(-alpha/beta, r2, y, N); 
        temp = r1; r1 = r2; r2 = y; y = temp;
        saddle_precond_solve(&paramsM1, &paramsM2, &system, r2, y, 0); 
        betaPrev = beta; 
        beta     = sqrt(dot_prod(r2, y, N)); 

        // apply previous rotation Qk-1 to get
        //   [delta_k eps_k+1 ] = [cs  sn][dbar_k   0       ]
        //   [gbar_k  dbar_k+1]   [sn -cs][alpha_k  beta_k+1].
        epsPrev = eps;
        delta   = cs*dbar + sn*alpha;
        gbar    = sn*dbar - cs*alpha;
        eps     =           sn*beta;
        dbar    =         - cs*beta;

        // compute the next plane rotation Qk
        gamma   = fmax(norm2_1d(gbar, beta), eps);
        cs      = gbar/gamma;
        sn      = beta/gamma;
        phi     = cs*phibar;
        phibar  = sn*phibar;

        // update x and rnorm
        temp = w1; w1 = w2; w2 = w; w = temp;
        axpy(-epsPrev, w1, v, N); 
        axpy(-delta , w2, v, N); 
        scalar_prod(1/gamma, v, w, N); 
        axpy(phi, w, x, N); 
        rnorm = phibar;

        // check stopping condition
        printf("  Iteration = %d; Approximated relative rnorm = %e\n", itn, rnorm/bnorm);
        if (rnorm <= rtol*bnorm) {
            printf("    Approximated relative rnorm (%e) satisfies rtol.\n", rnorm/bnorm);
            saddle_matvec_prod(&paramsA, &paramsB, &system, x, v, 0);
            axpy(-1.0, system.b, v, N); 
            rnormk = norm2(v, N);  
            if (rnormk <= rtol0*bnorm) {
                printf("    True relative rnorm (%e) satisfies rtol0.\n", rnormk/bnorm);
                break;  
            } else {
                printf("    True relative rnorm (%e) does NOT satisfiy rtol0.\n", rnormk/bnorm);
                rtol = rtol / 10;
            }
        }
    }
    getrusage(RUSAGE_SELF, &rtime1); // record end time
    double time = (rtime1.ru_utime.tv_sec - rtime0.ru_utime.tv_sec) + 1e-6*(rtime1.ru_utime.tv_usec - rtime0.ru_utime.tv_usec);  
    printf("Execution time: %f seconds.\n", time);
    printf("-----------------------------\n");

    // free allocated memory
    free(x); free(y); free(r1); free(r2);
    free(v); free(w); free(w1); free(w2);
    free_matvec_params(&paramsA, "A", 0, print);
    free_matvec_params(&paramsB, "B", 1, print);
    free_solver_params(&paramsM1, "M1", print); 
    free_solver_params(&paramsM2, "M2", print);
    free_saddle_point_system(&system, print);
}

/*
    
*/
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s matFilePath print\n", argv[0]);
        printf("    matFilePath: path to .mat file with respect to curr dir\n");
        printf("    print: includes additional print statements if set to 1\n");
        return 1;
    }
    char *file = argv[1]; int print = atoi(argv[2]);
    minres(file, print);
    return 0;
}
