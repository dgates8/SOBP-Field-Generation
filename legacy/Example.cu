sum(int *a, int*b, int*c, int n){
    int i = blockIdx.x*blockDim.x +threadIdx.x;
    if(i < n){
        c[i] = a[i] + b[i];
    }
}

sum(int *a, int*b, int*c, int n){
    int i = blockIdx.x*blockDim.x*2 +threadIdx.x;
    if(i < n){
        c[i] = a[i]*b[i];
        c[i + blockIdx.x] = a[i + blockIdx.x] + b[i + blockIdx.x];
    }
}

sum(int *a, int*b, int*c, int n){
    int i = blockIdx.x*blockDim.x*4 +threadIdx.x;
    if(i < n){
        #pragme unroll
        for(int k = 0; k < 4; k++){
            c[i + blockIdx.x*k] = a[i + blockIdx.x*k] + b[i + blockIdx.x*k];
        }
    }
}
