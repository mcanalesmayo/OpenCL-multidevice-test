__kernel void kernel_vect_add(__global int* restrict a, __global int* restrict b, int vlen, int which_device)
{
	printf("[%d] Got job from the queue\n", which_device);
	int i;
	for(i=0; i<vlen; i++){
		b[i] = i;
		a[i] = i;
	}

	for(i=0; i<vlen; i++) a[i] = a[i] + b[i];
	
	return;
}