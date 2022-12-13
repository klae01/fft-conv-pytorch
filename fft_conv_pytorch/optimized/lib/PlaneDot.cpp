#include <ATen/ATen.h>
#include <torch/extension.h>

#include <stdio.h>
#include <algorithm>

typedef unsigned int uint;

template <typename scalar_t>
void PlaneDot(
    const at::TensorAccessor<scalar_t, 5> AA,
    const at::TensorAccessor<scalar_t, 4> AB,
    at::TensorAccessor<scalar_t, 5> AC,
    const uint end_index)
{
    // index: bgosl
    // bgisl,goil->bgosl
    // sum over i
    uint i;
    uint sub_index[5];
    uint &tmp=sub_index[0], &B=sub_index[0], &G=sub_index[1], &O=sub_index[2], &S=sub_index[3], &L=sub_index[4];
    uint I=AA.size(2);
    for(uint index=0;index<end_index; index++)
    {
        for(tmp=index, i=5; --i; sub_index[i]=tmp%AC.size(i), tmp/=AC.size(i));
        AC[B][G][O][S][L]=0;
        for(i=0;i<I;i++)
            AC[B][G][O][S][L] += AA[B][G][i][S][L] * AB[G][O][i][L];
    }
}
template <typename scalar_t>
void PlaneDot_backprop_A(
    at::TensorAccessor<scalar_t, 5> AA,
    const at::TensorAccessor<scalar_t, 4> AB,
    const at::TensorAccessor<scalar_t, 5> AC,
    const uint end_index)
{
    // index: bgisl
    // bgisl,goil->bgosl
    // sum over o
    uint i;
    uint sub_index[5];
    uint &tmp=sub_index[0], &B=sub_index[0], &G=sub_index[1], &I=sub_index[2], &S=sub_index[3], &L=sub_index[4];
    uint O=AC.size(2);
    for(uint index=0;index<end_index; index++)
    {
        for(tmp=index, i=5; --i; sub_index[i]=tmp%AA.size(i), tmp/=AA.size(i));
        AA[B][G][I][S][L]=0;
        for(i=0;i<O;i++)
            AA[B][G][I][S][L] += AC[B][G][i][S][L] * AB[G][i][I][L];
    }
}
template <typename scalar_t>
void PlaneDot_backprop_B(
    const at::TensorAccessor<scalar_t, 5> AA,
    at::TensorAccessor<scalar_t, 4> AB,
    const at::TensorAccessor<scalar_t, 5> AC,
    const uint end_index)
{
    // index: goil
    // bgisl,goil->bgosl
    // sum over o
    uint i, j;
    uint sub_index[4];
    uint &tmp=sub_index[0], &G=sub_index[0], &O=sub_index[1], &I=sub_index[2], &L=sub_index[3];
    uint B=AC.size(0), S=AC.size(3);
    for(uint index=0;index<end_index; index++)
    {
        for(tmp=index, i=4; --i; sub_index[i]=tmp%AB.size(i), tmp/=AB.size(i));
        AB[G][O][I][L]=0;
        for(i=0;i<B;i++)
            for(j=0;j<S;j++)
                AB[G][O][I][L] += AA[i][G][I][j][L] * AC[i][G][O][j][L];
    }
}
void PlaneDot_wrapper(
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
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, Mat_A.type(), "PlaneDot", ([&] {
        PlaneDot<scalar_t> (
            Mat_A.accessor<scalar_t,5>(),
            Mat_B.accessor<scalar_t,4>(),
            Mat_C.accessor<scalar_t,5>(),
            Mat_C.numel()
        );
    }));
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
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, Mat_A.type(), "PlaneDot_BP_A", ([&] {
        PlaneDot_backprop_A<scalar_t> (
            Mat_A.accessor<scalar_t,5>(),
            Mat_B.accessor<scalar_t,4>(),
            Mat_C.accessor<scalar_t,5>(),
            Mat_A.numel()
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
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, Mat_A.type(), "PlaneDot_BP_B", ([&] {
        PlaneDot_backprop_B<scalar_t> (
            Mat_A.accessor<scalar_t,5>(),
            Mat_B.accessor<scalar_t,4>(),
            Mat_C.accessor<scalar_t,5>(),
            Mat_B.numel()
        );
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("PlaneDot_forward", &PlaneDot_wrapper, "PlaneDot");
    m.def("PlaneDot_backprop_A", &PlaneDot_backprop_A_wrapper, "PlaneDot backpropagation A");
    m.def("PlaneDot_backprop_B", &PlaneDot_backprop_B_wrapper, "PlaneDot backpropagation B");
}
