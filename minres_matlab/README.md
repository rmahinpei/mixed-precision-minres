# MATLAB Implementation
### [```minres_mixed.m```](https://github.com/rmahinpei/mixed-precision-minres/blob/main/minres_matlab/minres_mixed.m)

Source code is based off of [Michael Saunder's original MINRES implementation](https://web.stanford.edu/group/SOL/software/minres/) but has been adapted to support 
saddle-point systems and mixed precision matrix-vector products and preconditioner solves,

**Inputs**
* ``A``: $n \times n$ symmetric matrix;
* ``B``: $m \times n$ matrix with rank($B$) = $m$;
* ``L1``: $n \times n$ lower triangular/Cholesky factor of the $M_1$ preconditioner;
* ``U1``: $n \times n$ upper triangular/Cholesky factor of the $M_1$ preconditioner;
* ``L2``: $m \times m$ lower triangular/Cholesky factor of the $M_2$ preconditioner;
* ``U2``: $m \times m$ upper triangular/Choleskt factor of the $M_2$ preconditioner;
* ``f``: $n \times 1$ right-hand-side vector;
* ``g``: $m \times 1$ right-hand-side vector;
* ``fp32_prod``: whether matvec prod should be carried out in single precision;
* ``fp32_solve``: whether precond solves should be carried out in single precision.

**Outputs**

* ``x``: final estimate of the solution to the saddle-point system;
* ``relres_approx``: approximated relative residual of the precondtioned system for all iterations;
* ``relres_true``: true relative residual for all iterations;
* ``itn``: number of iterations taken by algorithm until termination;
* ``dot_prods``: vector of dot products between the two most recent Lanczos vectors at each iteration.

### [```minres_script.m```](https://github.com/rmahinpei/mixed-precision-minres/blob/main/minres_matlab/minres_script.m)

Driver code for [```minres_mixed.m```](https://github.com/rmahinpei/mixed-precision-minres/blob/main/minres_matlab/minres_mixed.m).
