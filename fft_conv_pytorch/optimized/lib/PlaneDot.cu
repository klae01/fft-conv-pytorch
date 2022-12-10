#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <stdio.h>
#include <algorithm>

typedef unsigned int uint;

template <typename scalar_t>
__global__ void PlaneDot(
    const at::PackedTensorAccessor32<scalar_t, 5, at::RestrictPtrTraits> AA,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> AB,
    at::PackedTensorAccessor32<scalar_t, 5, at::RestrictPtrTraits> AC,
    const uint end_index, const uint jump)
{
    // index: bgosl
    // bgisl,goil->bgosl
    // sum over i
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    uint i;
    uint sub_index[5];
    uint &tmp=sub_index[0], &B=sub_index[0], &G=sub_index[1], &O=sub_index[2], &S=sub_index[3], &L=sub_index[4];
    uint I=AA.size(2);
    for(;index<end_index; index+=jump)
    {
        for(tmp=index, i=5; --i; sub_index[i]=tmp%AC.size(i), tmp/=AC.size(i));
        AC[B][G][O][S][L]=0;
        for(i=0;i<I;i++)
            AC[B][G][O][S][L] += AA[B][G][i][S][L] * AB[G][O][i][L];
    }
}
template <typename scalar_t>
__global__ void PlaneDot_backprop_A(
    at::PackedTensorAccessor32<scalar_t, 5, at::RestrictPtrTraits> AA,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> AB,
    const at::PackedTensorAccessor32<scalar_t, 5, at::RestrictPtrTraits> AC,
    const uint end_index, const uint jump)
{
    // index: bgisl
    // bgisl,goil->bgosl
    // sum over o
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    uint i;
    uint sub_index[5];
    uint &tmp=sub_index[0], &B=sub_index[0], &G=sub_index[1], &I=sub_index[2], &S=sub_index[3], &L=sub_index[4];
    uint O=AC.size(2);
    for(;index<end_index; index+=jump)
    {
        for(tmp=index, i=5; --i; sub_index[i]=tmp%AA.size(i), tmp/=AA.size(i));
        AA[B][G][I][S][L]=0;
        for(i=0;i<O;i++)
            AA[B][G][I][S][L] += AC[B][G][i][S][L] * AB[G][i][I][L];
    }
}
template <typename scalar_t>
__global__ void PlaneDot_backprop_B(
    const at::PackedTensorAccessor32<scalar_t, 5, at::RestrictPtrTraits> AA,
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> AB,
    const at::PackedTensorAccessor32<scalar_t, 5, at::RestrictPtrTraits> AC,
    const uint end_index, const uint jump)
{
    // index: goil
    // bgisl,goil->bgosl
    // sum over o
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    uint i, j;
    uint sub_index[4];
    uint &tmp=sub_index[0], &G=sub_index[0], &O=sub_index[1], &I=sub_index[2], &L=sub_index[3];
    uint B=AC.size(0), S=AC.size(3);
    for(;index<end_index; index+=jump)
    {
        for(tmp=index, i=4; --i; sub_index[i]=tmp%AB.size(i), tmp/=AB.size(i));
        AB[G][O][I][L]=0;
        for(i=0;i<B;i++)
            for(j=0;j<S;j++)
                AB[G][O][I][L] += AA[i][G][I][j][L] * AC[i][G][O][j][L];
    }
}
void determin_task_divide(uint works, uint device, uint&nBlock, uint&nThread) {
    cudaDeviceProp *prop=at::cuda::getDeviceProperties(device);
    works = (works + 31) / 32;
    nBlock = prop->multiProcessorCount * prop->maxBlocksPerMultiProcessor;
    nThread = prop->maxThreadsPerBlock / 32;
    nBlock = std::min(nBlock, (works + nThread - 1) / nThread);
    nThread = std::min(nThread, (works + nBlock - 1) / nBlock);
    nThread *= 32;
    // printf("reduce: nblock: %d \t nThread: %d\t per thread: %.2f\n", nBlock, nThread, (float)works/nBlock/nThread);
}
at::Tensor PlaneDot_wrapper(
    at::Tensor Mat_A,
    at::Tensor Mat_B
) {
    uint B=Mat_A.sizes()[0];
    uint G=Mat_A.sizes()[1];
    uint I=Mat_A.sizes()[2];
    uint S=Mat_A.sizes()[3];
    uint L=Mat_A.sizes()[4];
    uint O=Mat_B.sizes()[1];
    at::TensorOptions Topt;
    Topt = Topt.device(Mat_A.device()).dtype(Mat_A.dtype());
    at::Tensor Mat_C=at::empty({B, G, O, S, L}, Topt);
    uint nBlock, nThread;
    determin_task_divide(Mat_C.numel(), (uint)Mat_A.device().index(), nBlock, nThread);
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, Mat_A.type(), "PlaneDot", ([&] {
        PlaneDot<scalar_t> <<<nBlock,nThread,0,at::cuda::getCurrentCUDAStream()>>>(
            Mat_A.packed_accessor32<scalar_t,5,at::RestrictPtrTraits>(),
            Mat_B.packed_accessor32<scalar_t,4,at::RestrictPtrTraits>(),
            Mat_C.packed_accessor32<scalar_t,5,at::RestrictPtrTraits>(),
            Mat_C.numel(),
            nBlock*nThread
        );
    }));
    return Mat_C;
}
void PlaneDot_backprop_A_wrapper(
    at::Tensor Mat_A,
    at::Tensor Mat_B,
    at::Tensor Mat_C
) {
    uint B=Mat_A.sizes()[0];
    uint G=Mat_A.sizes()[1];
    uint I=Mat_A.sizes()[2];
    uint S=Mat_A.sizes()[3];
    uint L=Mat_A.sizes()[4];
    uint O=Mat_B.sizes()[1];
    uint nBlock, nThread;
    determin_task_divide(Mat_A.numel(), (uint)Mat_A.device().index(), nBlock, nThread);
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, Mat_A.type(), "PlaneDot_BP_A", ([&] {
        PlaneDot_backprop_A<scalar_t> <<<nBlock,nThread,0,at::cuda::getCurrentCUDAStream()>>>(
            Mat_A.packed_accessor32<scalar_t,5,at::RestrictPtrTraits>(),
            Mat_B.packed_accessor32<scalar_t,4,at::RestrictPtrTraits>(),
            Mat_C.packed_accessor32<scalar_t,5,at::RestrictPtrTraits>(),
            Mat_A.numel(),
            nBlock*nThread
        );
    }));
}
void PlaneDot_backprop_B_wrapper(
    at::Tensor Mat_A,
    at::Tensor Mat_B,
    at::Tensor Mat_C
) {
    uint B=Mat_A.sizes()[0];
    uint G=Mat_A.sizes()[1];
    uint I=Mat_A.sizes()[2];
    uint S=Mat_A.sizes()[3];
    uint L=Mat_A.sizes()[4];
    uint O=Mat_B.sizes()[1];
    uint nBlock, nThread;
    determin_task_divide(Mat_B.numel(), (uint)Mat_A.device().index(), nBlock, nThread);
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, Mat_A.type(), "PlaneDot_BP_B", ([&] {
        PlaneDot_backprop_B<scalar_t> <<<nBlock,nThread,0,at::cuda::getCurrentCUDAStream()>>>(
            Mat_A.packed_accessor32<scalar_t,5,at::RestrictPtrTraits>(),
            Mat_B.packed_accessor32<scalar_t,4,at::RestrictPtrTraits>(),
            Mat_C.packed_accessor32<scalar_t,5,at::RestrictPtrTraits>(),
            Mat_B.numel(),
            nBlock*nThread
        );
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("PlaneDot_forward", &PlaneDot_wrapper, "PlaneDot CUDA");
    m.def("PlaneDot_backprop_A", &PlaneDot_backprop_A_wrapper, "PlaneDot backpropagation A CUDA");
    m.def("PlaneDot_backprop_B", &PlaneDot_backprop_B_wrapper, "PlaneDot backpropagation B CUDA");
}
