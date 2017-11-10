#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include "test.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#define ALIGNMENT 64

#ifdef WIN
	#include <windows.h>
#else
	#include <pthread.h>
	#include <sys/time.h>
	double gettime() {
		struct timeval t;
		gettimeofday(&t,NULL);
		return t.tv_sec+t.tv_usec*1e-6;
	}
#endif


#ifdef NV
	#include <oclUtils.h>
#else
	#include <CL/cl.h>
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 512
#endif


// local variables
static cl_platform_id  *platform_ids;
static cl_context	    context1;
static cl_context	    context2;
static cl_command_queue cmd_queue1;
static cl_command_queue cmd_queue2;
static cl_command_queue cmd_queue3;
static cl_device_id    *device_list1;
static cl_device_id    *device_list2;
static cl_int           num_devices;

static int initialize()
{
	cl_int result;
	size_t size;

	// create OpenCL context

	cl_platform_id platform_id;
	cl_uint num_platforms;
	cl_platform_info platform_info;
	char platform_name[30];
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(0,NULL,&num_platforms) failed\n"); return -1; }
	printf("Number of platforms: %d\n", num_platforms);
	platform_ids = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
	if (clGetPlatformIDs(num_platforms, platform_ids, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(num_platforms,platform_ids,NULL) failed\n"); return -1; }

	for(int i=0; i<num_platforms; i++){
		clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 30, platform_name, NULL);
		printf("Platform %d: %s\n", i, platform_name);
	}

	// Intel Altera is idx=1
	// cl_context_properties:
	// Specifies a list of context property names and their corresponding values. Each property name is immediately followed by the corresponding desired value.
	// The list is terminated with 0. properties can be NULL in which case the platform that is selected is implementation-defined.
	// The list of supported properties is described in the table below.
	cl_context_properties ctxprop_fpga[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform_ids[1], 0};

	context1 = clCreateContextFromType(ctxprop_fpga, CL_DEVICE_TYPE_ACCELERATOR, NULL, NULL, NULL);
	if( !context1 ) { printf("ERROR: clCreateContextFromType(%s) failed\n", "FPGA"); return -1; }

	// get the list of FPGAs
	result = clGetContextInfo(context1, CL_CONTEXT_DEVICES, 0, NULL, &size);
	num_devices = (int) (size / sizeof(cl_device_id));
	
	if( result != CL_SUCCESS || num_devices < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list1 = new cl_device_id[num_devices];
	if( !device_list1 ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
	result = clGetContextInfo( context1, CL_CONTEXT_DEVICES, size, device_list1, NULL );
	if( result != CL_SUCCESS ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }

	// create command queue for the first device
	cmd_queue1 = clCreateCommandQueue(context1, device_list1[0], 0, NULL);
	if( !cmd_queue1 ) { printf("ERROR: clCreateCommandQueue() FPGA failed\n"); return -1; }



	// NVIDIA CUDA is idx=0
	cl_context_properties ctxprop_gpu[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform_ids[0], 0};
	context2 = clCreateContextFromType(ctxprop_gpu, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL );
	if( !context2 ) { printf("ERROR: clCreateContextFromType(%s) failed\n", "GPU"); return -1; }

	// get the list of GPUs
	result = clGetContextInfo( context2, CL_CONTEXT_DEVICES, 0, NULL, &size );
	num_devices = (int) (size / sizeof(cl_device_id));
	
	if( result != CL_SUCCESS || num_devices < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list2 = new cl_device_id[num_devices];
	if( !device_list2 ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
	result = clGetContextInfo( context2, CL_CONTEXT_DEVICES, size, device_list2, NULL );
	if( result != CL_SUCCESS ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }

	// create command queue for the first device
	cmd_queue2 = clCreateCommandQueue(context2, device_list2[0], 0, NULL);
	if( !cmd_queue2 ) { printf("ERROR: clCreateCommandQueue() GPU 1 failed\n"); return -1; }
	cmd_queue3 = clCreateCommandQueue(context2, device_list2[1], 0, NULL);
	if( !cmd_queue3 ) { printf("ERROR: clCreateCommandQueue() GPU 2 failed\n"); return -1; }

	return 0;
}

static int shutdown()
{
	// release resources
	if( cmd_queue1 ) clReleaseCommandQueue( cmd_queue1 );
	if( cmd_queue2 ) clReleaseCommandQueue( cmd_queue2 );
	if( cmd_queue3 ) clReleaseCommandQueue( cmd_queue3 );
	if( context1 ) clReleaseContext( context1 );
	if( context2 ) clReleaseContext( context2 );
	if( device_list1 ) delete device_list1;
	if( device_list2 ) delete device_list2;

	free(platform_ids);

	// reset all variables
	cmd_queue1 = 0;
	cmd_queue2 = 0;
	cmd_queue3 = 0;
	context1 = 0;
	context2 = 0;
	device_list1 = 0;
	device_list2 = 0;
	num_devices = 0;

	return 0;
}

cl_mem fpga_vect_a;
cl_mem fpga_vect_b;
cl_mem gpu_vect_a;
cl_mem gpu_vect_b;

cl_int *f_vect_a;
cl_int *f_vect_b;
cl_int *g_vect_a;
cl_int *g_vect_b;

cl_kernel fpga_kernel_vect_add;
cl_kernel gpu_kernel_vect_add;

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) {
   cl_ulong a;
   clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
   printf("%-40s = %lu\n", name, a);
}
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name) {
   cl_uint a;
   clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
   printf("%-40s = %u\n", name, a);
}
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name) {
   cl_bool a;
   clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
   printf("%-40s = %s\n", name, (a?"true":"false"));
}
static void device_info_string( cl_device_id device, cl_device_info param, const char* name) {
   char a[1024];
   clGetDeviceInfo(device, param, 1024, &a, NULL);
   printf("%-40s = %s\n", name, a);
}

static void display_device_info( cl_device_id device ) {

   printf("Querying device for info:\n");
   printf("========================\n");
   device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
   device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
   device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
   device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
   device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
   device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
   device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
   device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
   device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
   device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
   device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
   device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
   device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
   device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

   {
      cl_command_queue_properties ccp;
      clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
      printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
      printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
   }
}

int all_setup()
{

	int sourcesize = 1024*1024;
	char *source = (char *) calloc(sourcesize, sizeof(char)); 
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	// read the kernel core source
	char *tempchar = "./test.cl";
	FILE *fp = fopen(tempchar, "rb"); 
	if(!fp) { printf("ERROR: unable to open '%s'\n", tempchar); return -1; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);
		
	// OpenCL initialization
	if(initialize()) { printf("ERROR: initializing\n"); return -1;}

	/*printf("clBuildProgram errors: %d %d %d %d %d %d %d %d %d %d\n",
		CL_INVALID_PROGRAM, CL_INVALID_VALUE, CL_INVALID_DEVICE, CL_INVALID_BINARY, CL_INVALID_BUILD_OPTIONS, CL_INVALID_OPERATION,
		CL_COMPILER_NOT_AVAILABLE, CL_BUILD_PROGRAM_FAILURE, CL_INVALID_OPERATION, CL_OUT_OF_HOST_MEMORY);*/
	/*printf("clCreateKernel errors: %d %d %d %d %d %d %d %d %d %d\n",
		CL_INVALID_PROGRAM, CL_INVALID_PROGRAM_EXECUTABLE, CL_INVALID_KERNEL_NAME, CL_INVALID_KERNEL_DEFINITION, CL_INVALID_VALUE, CL_OUT_OF_HOST_MEMORY);*/

	// compile kernel
	cl_int err = 0;
	const char *slist[2] = { source, 0 };

	/* ------------ */
	/* FPGA program */
	/* ------------ */

	// Query the available OpenCL devices.
	cl_device_id *devices;
	cl_uint num_devices;

	devices = aocl_utils::getDevices(platform_ids[1], CL_DEVICE_TYPE_ALL, &num_devices);

	// We'll just use the first device.
	cl_device_id device = devices[0];
	// Display some device information.
  	//display_device_info(device);

	// Create the FPGA program.
  	std::string binary_file = aocl_utils::getBoardBinaryFile("/home/mcanales/socarrat-test/test", device);
  	printf("Using AOCX: %s\n", binary_file.c_str());
  	cl_program prog1 = aocl_utils::createProgramFromBinary(context1, binary_file.c_str(), &device, 1);
  	err = clBuildProgram(prog1, 0, NULL, NULL, NULL, NULL);
  	if (err != CL_SUCCESS) { printf("ERROR: FPGA clBuildProgram() => %d\n", err); return -1; }

  /* ----------- */
	/* GPU program */
	/* ----------- */

	cl_program prog2 = clCreateProgramWithSource(context2, 1, slist, NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: GPU clCreateProgramWithSource() => %d\n", err); return -1; }
	err = clBuildProgram(prog2, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS) {
		printf("ERROR: GPU clBuildProgram() => %d\n", err);
		if (err == CL_BUILD_PROGRAM_FAILURE) {
		    // Determine the size of the log
		    size_t log_size;
		    devices = aocl_utils::getDevices(platform_ids[0], CL_DEVICE_TYPE_ALL, &num_devices);
		    clGetProgramBuildInfo(prog2, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		    // Allocate memory for the log
		    char *log = (char *) malloc(log_size);
		    // Get the log
		    clGetProgramBuildInfo(prog2, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		    // Print the log
		    printf("%s\n", log);



		    clGetProgramBuildInfo(prog2, devices[1], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		    log = (char *) realloc(log, log_size);
		    // Get the log
		    clGetProgramBuildInfo(prog2, devices[1], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		    // Print the log
		    printf("%s\n", log);

		    free(log);
		}

		return -1;
	}
	
	char *kernel_vect_add = "kernel_vect_add";
		
	fpga_kernel_vect_add = clCreateKernel(prog1, kernel_vect_add, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: FPGA clCreateKernel() 0 => %d\n", err); return -1; }
	gpu_kernel_vect_add = clCreateKernel(prog2, kernel_vect_add, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: GPU clCreateKernel() 0 => %d\n", err); return -1; }

	clReleaseProgram(prog1);
	clReleaseProgram(prog2);
	
	cl_int vlen = 4096;
	f_vect_a = (cl_int *) memalign(ALIGNMENT, vlen * sizeof(cl_int));
	f_vect_b = (cl_int *) memalign(ALIGNMENT, vlen * sizeof(cl_int));
	g_vect_a = (cl_int *) memalign(ALIGNMENT, vlen * sizeof(cl_int));
	g_vect_b = (cl_int *) memalign(ALIGNMENT, vlen * sizeof(cl_int));
	fpga_vect_a = clCreateBuffer(context1, CL_MEM_READ_WRITE, vlen * sizeof(cl_int), NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer fpga_vect_a (size:%d) => %d\n", vlen, err); return -1;}
	fpga_vect_b = clCreateBuffer(context1, CL_MEM_READ_WRITE, vlen * sizeof(cl_int), NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer fpga_vect_b (size:%d) => %d\n", vlen, err); return -1;}
	gpu_vect_a = clCreateBuffer(context2, CL_MEM_READ_WRITE, vlen * sizeof(cl_int), NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer gpu_vect_a (size:%d) => %d\n", vlen, err); return -1;}
	gpu_vect_b = clCreateBuffer(context2, CL_MEM_READ_WRITE, vlen * sizeof(cl_int), NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer gpu_vect_b (size:%d) => %d\n", vlen, err); return -1;}
	
	//write buffers
	err = clEnqueueWriteBuffer(cmd_queue1, fpga_vect_a, CL_TRUE, 0, vlen * sizeof(cl_int), f_vect_a, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: FPGA clEnqueueWriteBuffer vect_a (size:%d) => %d\n", vlen * sizeof(cl_int), err); return -1; }

	err = clEnqueueWriteBuffer(cmd_queue1, fpga_vect_b, CL_TRUE, 0, vlen * sizeof(cl_int), f_vect_b, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: FPGA clEnqueueWriteBuffer vect_b (size:%d) => %d\n", vlen * sizeof(cl_int), err); return -1; }

	err = clEnqueueWriteBuffer(cmd_queue2, gpu_vect_a, CL_TRUE, 0, vlen * sizeof(cl_int), g_vect_a, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU clEnqueueWriteBuffer vect_a (size:%d) => %d\n", vlen * sizeof(cl_int), err); return -1; }

	err = clEnqueueWriteBuffer(cmd_queue2, gpu_vect_b, CL_TRUE, 0, vlen * sizeof(cl_int), g_vect_b, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU clEnqueueWriteBuffer vect_b (size:%d) => %d\n", vlen * sizeof(cl_int), err); return -1; }

	err = clEnqueueWriteBuffer(cmd_queue3, gpu_vect_a, CL_TRUE, 0, vlen * sizeof(cl_int), g_vect_a, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU clEnqueueWriteBuffer vect_a (size:%d) => %d\n", vlen * sizeof(cl_int), err); return -1; }

	err = clEnqueueWriteBuffer(cmd_queue3, gpu_vect_b, CL_TRUE, 0, vlen * sizeof(cl_int), g_vect_b, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU clEnqueueWriteBuffer vect_b (size:%d) => %d\n", vlen * sizeof(cl_int), err); return -1; }

	cl_int dev_fpga = 0;
	cl_int dev_gpu = 1;
	clSetKernelArg(fpga_kernel_vect_add, 0, sizeof(cl_mem), (void *) &fpga_vect_a);
	clSetKernelArg(fpga_kernel_vect_add, 1, sizeof(cl_mem), (void *) &fpga_vect_b);
	clSetKernelArg(fpga_kernel_vect_add, 2, sizeof(cl_int), (void*) &vlen);
	clSetKernelArg(fpga_kernel_vect_add, 3, sizeof(cl_int), (void*) &dev_fpga);
	clSetKernelArg(gpu_kernel_vect_add, 0, sizeof(cl_mem), (void *) &gpu_vect_a);
	clSetKernelArg(gpu_kernel_vect_add, 1, sizeof(cl_mem), (void *) &gpu_vect_a);
	clSetKernelArg(gpu_kernel_vect_add, 2, sizeof(cl_int), (void*) &vlen);
	clSetKernelArg(gpu_kernel_vect_add, 3, sizeof(cl_int), (void*) &dev_gpu);
	
	size_t global_work[3] = { vlen, 1, 1 };
	//size_t local_work_size = BLOCK_SIZE;

	err = clEnqueueNDRangeKernel(cmd_queue1, fpga_kernel_vect_add, CL_TRUE, NULL, global_work, NULL, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: FPGA clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

	clFinish(cmd_queue1);
	clEnqueueReadBuffer(cmd_queue1, fpga_vect_a, CL_TRUE, 0, vlen * sizeof(cl_int), f_vect_a, 0, 0, 0);
	for(int i=0; i<vlen; i++){
		printf("FPGA a[%d] = %d\n", i, f_vect_a[i]);
	}
	sleep(3);

	err = clEnqueueNDRangeKernel(cmd_queue2, gpu_kernel_vect_add, CL_TRUE, NULL, global_work, NULL, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

	clFinish(cmd_queue2);
	clEnqueueReadBuffer(cmd_queue2, gpu_vect_a, CL_TRUE, 0, vlen * sizeof(cl_int), g_vect_a, 0, 0, 0);
	for(int i=0; i<vlen; i++){
		printf("GPU0 a[%d] = %d\n", i, g_vect_a[i]);
	}
	sleep(3);

	err = clEnqueueNDRangeKernel(cmd_queue3, gpu_kernel_vect_add, CL_TRUE, NULL, global_work, NULL, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

	clFinish(cmd_queue3);
	clEnqueueReadBuffer(cmd_queue3, gpu_vect_a, CL_TRUE, 0, vlen * sizeof(cl_int), g_vect_a, 0, 0, 0);
	for(int i=0; i<vlen; i++){
		printf("GPU1 a[%d] = %d\n", i, g_vect_a[i]);
	}
	sleep(3);

}

void deallocateMemory()
{
	clReleaseMemObject(fpga_vect_a);
	clReleaseMemObject(fpga_vect_b);
	clReleaseMemObject(gpu_vect_a);
	clReleaseMemObject(gpu_vect_b);

	free(f_vect_a);
	free(f_vect_b);
	free(g_vect_a);
	free(g_vect_b);

	shutdown();
}

int main(int argc, char **argv)
{
	printf("WG size of fpga_kernel_vect_add = %d\n", BLOCK_SIZE);

	all_setup();
	deallocateMemory();
}