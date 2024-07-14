# CUDA C Implementation (Chol)
Note that this implementation expects the **Cholesky factors** of the preconditioners $M_1$ and $M_2$ within the input .mat file.

### Set-Up and Build Instructions
To successfully compile our source code, please make sure that you have all the required tools and libraries installed:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [MATIO Library](https://github.com/tbeu/matio/tree/master)
- [HDF5 library](https://www.hdfgroup.org/solutions/hdf5/) (required by MATIO)
- [Zlib library](https://www.zlib.net/) (required by MATIO)

You can then set the paths to the required include and library directories in the provided Makefile and compile the source code using the command
```
make minres
```
To run the executable, use the command
```
minres PATH-TO-MAT-FILE PRINT-FLAG 
```
where:
* ``PATH-TO-MAT-FILE`` is the path to the .mat file (with respect to the curr dir) the contains the saddle-point system of interest;
* ``PRINT-FLAG`` is the boolean flag indicating whether additional print statements should be included or not.

For example, if we intend to use ``../matrices/test.mat`` as our saddle-point system and want the additional print statements, we would use the command:
```
minres ../matrices/test.mat 1
```
