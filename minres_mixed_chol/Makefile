# To customize the build, you must specify the following variables:
#   NVCC    - Path to the NVCC compiler
#   INC     - Include directories for the CUDA Toolkit and MATIO
#   L_DIR   - Library directories for the CUDA Toolkit and MATIO
#
# Example usage:
#   make NVCC=/path/to/nvcc INC="-I/path/to/include -I../matio/src" L_DIR="-L/path/to/lib64 -L../matio"

# Path to the NVCC compiler
NVCC ?= 
# Include directories
INC ?= 
# Library directories
L_DIR ?= 

CUFLAGS := -arch=sm_86
LIBS 	:= -lcusparse -lcublas -lmatio  

all: minres

minres: minres.c
	$(NVCC) $(CUFLAGS) $(INC) utils.c minres.c -o minres $(L_DIR) $(LIBS)

clean:
	rm -f minres