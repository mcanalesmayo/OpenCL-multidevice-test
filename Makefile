include make.config
# C compiler
CC = g++
CC_FLAGS = -g -O2 
# CUDA
CUDA_OCL_LIB=/usr/local/cuda/lib64
CUDA_OCL_INC=/usr/local/cuda/include

AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := $(shell aocl link-config )
ALTERA_LIB_DIRS=
ALTERA_LIBS=rt
ALTERA_INC_DIRS := /home/mcanales/socarrat-test/inc
ALTERA_SRCS := /home/mcanales/socarrat-test/src/AOCLUtils/*.cpp

test: clean
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL test.cpp $(ALTERA_SRCS) -o test.o $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) -I$(CUDA_OCL_INC) $(foreach D,$(ALTERA_INC_DIRS),-I$D) -L$(CUDA_OCL_LIB) $(foreach D,$(ALTERA_LIB_DIRS),-L$D) $(foreach L,$(ALTERA_LIBS),-l$L)

clean:
	rm -f *.o
