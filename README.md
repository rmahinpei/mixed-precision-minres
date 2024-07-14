# Mixed Precision Minimial Residual (MINRES) Method
## Motivation
Within the scientific computing community, there is a growing interest in using lower precision formats, such as single and half precision, given the potential for increased speed, reduced energy, and reduced communication costs compared to the traditional double precision format. Mixed precision algorithms, which combine different precisions, offer a promising avenue to balance speed and accuracy. However, the applicability of mixed precision algorithms varies across different problem domains, thus requiring domain-specific investigations. [This thesis](https://github.com/rmahinpei/mixed-precision-minres/blob/main/thesis_final.pdf) addresses the gap in the domain of sparse preconditioned saddle-point problems, proposing two mixed precision variants of the Minimal Residual (MINRES) method tailored to varying the arithmetic and storage precision of the preconditioner solves and the matrix-vector products. Using CUDA C, these variants are implemented and then used to make performance measurements on NVIDIA's GeForce RTX 3070 Ti graphics card for Maxwell and Stokes saddle-point problems. Our numerical results confirm that low precision preconditioning is an effective optimization strategy for reducing the runtime of the MINRES iterative solver when applied to the chosen saddle-point problems while maintaining the desired accuracy of the final solution.  

## Implementations
We provide both the CUDA C and MATLAB implementation of our proposed mixed precision MINRES scheme. Both implementations solve a saddle point system of the form 

$$\begin{bmatrix} A & B^T \\\ B & 0\end{bmatrix} \begin{bmatrix}x \\\ y\end{bmatrix} = \begin{bmatrix}f \\\ g\end{bmatrix}$$

preconditioned by a block diagonal preconditioner of the form

$$\begin{bmatrix}M_1 & 0 \\\ 0 & M_2 \end{bmatrix}$$

The [``minres_mixed_chol``](https://github.com/rmahinpei/mixed-precision-minres/tree/main/minres_mixed_chol) directory provides the CUDA C source code that expects the **Cholesky factors** of the preconditioners $M_1$ and $M_2$ as input. The [``minres_matlab``](https://github.com/rmahinpei/mixed-precision-minres/tree/main/minres_matlab) directory provides the MATLAB source code that accepts both the Cholesky and LU factors of those preconditioners as input.

-------
*This thesis was submitted in partial fulfillment of the requirements for the degree of Bachelor of Science (Honours) in Computer Science at the University of British Columbia (UBC). The project was supervised by [Dr. Chen Greif](https://www.cs.ubc.ca/~greif/), whose steadfast support and insights were instrumental in its advancement.*  
