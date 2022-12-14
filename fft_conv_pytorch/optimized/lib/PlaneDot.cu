#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <stdio.h>
#include <algorithm>
#include <cassert>


#define HOST_INIT(TYPE, OUT, LIMIT, ...)                                                \
{                                                                                       \
    auto DataLoad = __VA_ARGS__;                                                        \
    OUT = (TYPE*)malloc(LIMIT * sizeof(TYPE));                                          \
    for(size_t __iteration=0; __iteration < LIMIT; __iteration++)                       \
        OUT[__iteration] = DataLoad(__iteration);                                       \
}
#define CUDA_INIT(TYPE, OUT, LIMIT, ...)                                                \
{                                                                                       \
    auto DataLoad = __VA_ARGS__;                                                        \
    cudaMalloc(&OUT, LIMIT * sizeof(TYPE));                                             \
    TYPE *__host_array=(TYPE*)malloc(LIMIT * sizeof(TYPE));                             \
    for(size_t __iteration=0; __iteration < LIMIT; __iteration++)                       \
        __host_array[__iteration] = DataLoad(__iteration);                              \
    cudaMemcpy(OUT, __host_array, LIMIT * sizeof(TYPE), cudaMemcpyHostToDevice);        \
}
#define TO_CUDA(TYPE, ARRAY, LIMIT)                                                     \
{                                                                                       \
    TYPE *__device_array;                                                               \
    cudaMalloc(&__device_array, LIMIT * sizeof(TYPE));                                  \
    cudaMemcpy(__device_array, ARRAY, LIMIT * sizeof(TYPE), cudaMemcpyHostToDevice);    \
    ARRAY = __device_array;                                                             \
}
#define ProblemInfoShortCut(INFO)   \
    size_t const &ndim=INFO.ndim;   \
    size_t *const &DIM=INFO.DIM;    \
    size_t *const &DIV=INFO.DIV;    \
    size_t *const &BLK=INFO.BLK;

template <typename scalar_t>
struct MatrixInfo {
    scalar_t *Mat;
    scalar_t *Fetch;
    scalar_t *prev_offset;
    size_t *stride;
    size_t ndim;
    size_t fetch_numel;
    size_t effective_dim;
    size_t last_Jobindex;
    MatrixInfo(at::Tensor &Matrix, size_t fetch_numel_, size_t effective_dim_) {
        Mat = Matrix.data_ptr<scalar_t>();
        ndim = Matrix.ndimension();
        fetch_numel = fetch_numel_;
        effective_dim = effective_dim_;
        assert(ndim == __builtin_popcountll(effective_dim));
        CUDA_INIT(size_t, stride, ndim, [&](size_t i){return Matrix.stride(i);});
    }
    __device__ scalar_t* initialize(scalar_t *memory, const size_t &offset)
    {
        prev_offset = nullptr;
        Fetch = memory + offset;
        return Fetch;
    }
};
struct ProblemInfo {
    size_t *DIM;
    size_t *DIV;
    size_t *BLK;
    size_t ndim;
    size_t NumJobs;     // equal to prod(BLK)
    size_t TaskSize;    // equal to prod(DIV)
    size_t JobChunk;    // equal to prod(<Non-shareable-job-continuous-dimension>)
    // DIM[0:ndim] = {I J K L M N O}
    // .......................<---> : reduction axis -> ChunkDims = 3
    // ...................<->       : broadcasting axis
    // ...............<->           : sharing axis
    // ...............<----->       : non reduction axis / axis of output

    template <typename scalar_t>
    ProblemInfo(
        const at::TensorAccessor<scalar_t, 1> _DIM,
        const at::TensorAccessor<scalar_t, 1> _DIV,
        size_t ChunkDims
    ) {
        assert(_DIM.size(0)==_DIV.size(0));
        assert(ChunkDims<_DIV.size(0));
        ndim=_DIM.size(0);
        HOST_INIT(size_t, DIM, ndim, [&](size_t i){return _DIM[i];});
        HOST_INIT(size_t, DIV, ndim, [&](size_t i){return _DIV[i];});
        HOST_INIT(size_t, BLK, ndim, [&](size_t i){return 1 + (_DIM[i] - 1) / _DIV[i];});
        NumJobs=1; TaskSize=1; JobChunk=1;
        for(size_t i=0; i<ndim; i++)        NumJobs *=BLK[i];
        for(size_t i=0; i<ndim; i++)        TaskSize*=DIV[i];
        for(size_t i=0; i<ChunkDims; i++)   JobChunk*=BLK[ndim-i-1];
        TO_CUDA(size_t, DIM, ndim);
        TO_CUDA(size_t, DIV, ndim);
        TO_CUDA(size_t, BLK, ndim);
    }
    __device__ size_t getJobIndex(size_t worker_id, size_t worker_pool)
    { return NumJobs / JobChunk * worker_id / worker_pool * JobChunk; }
    size_t DivisibleJobCount()
    { return NumJobs / JobChunk; }
};

template <typename scalar_t>
__device__ scalar_t* getBlockOffset(
    MatrixInfo<scalar_t> &INFO,
    const ProblemInfo &INFO_P,
    size_t index
) {
    ProblemInfoShortCut(INFO_P);
    size_t position=0, effective_dim=INFO.effective_dim;
    for(size_t i=ndim, lv=INFO.ndim; i--; index /= BLK[i], effective_dim >>= 1)
        if(effective_dim & 1)
            position += index % BLK[i] * INFO.stride[--lv] * DIV[i];
    return INFO.Mat + position;
}

template <typename scalar_t>
__device__ scalar_t* getDivisionRelativeIndex(
    MatrixInfo<scalar_t> &INFO,
    const ProblemInfo &INFO_P,
    size_t JOBindex,
    size_t DIVindex,
    scalar_t *&offset
) {
    // return nullptr if out-of-matrix condition
    ProblemInfoShortCut(INFO_P);
    size_t position=0, effective_dim=INFO.effective_dim;
    for(size_t i=ndim, lv=INFO.ndim; i--; JOBindex /= BLK[i], DIVindex /= DIV[i], effective_dim >>= 1)
        if(effective_dim & 1)
            if(DIM[i] > JOBindex % BLK[i] * DIV[i] + DIVindex % DIV[i])
                position += DIVindex % DIV[i] * INFO.stride[--lv];
            else
                return nullptr;
    return offset + position;
}

template <typename scalar_t>
__device__ size_t getFetchRelativeIndex(
    MatrixInfo<scalar_t> &INFO,
    const ProblemInfo &INFO_P,
    size_t index
) {
    ProblemInfoShortCut(INFO_P);
    size_t position=0, stride=1, effective_dim=INFO.effective_dim;
    for(size_t i=ndim; i--; index /= DIV[i], effective_dim >>= 1)
        if(effective_dim & 1)
        {
            position += index % DIV[i] * stride;
            stride *= DIV[i];
        }
    return position;
}

template <typename scalar_t>
__device__ void writebackMatrix(
    MatrixInfo<scalar_t> &INFO,
    const ProblemInfo &INFO_P,
    const size_t &Worker
) {
    scalar_t *pointer;
    if(INFO.prev_offset != nullptr)
        for(size_t DIVindex=0; DIVindex<INFO.fetch_numel; DIVindex+=Worker)
        {
            pointer = getDivisionRelativeIndex<scalar_t>(
                INFO, INFO_P, INFO.last_Jobindex, DIVindex, INFO.prev_offset
            );
            if(pointer != nullptr)
                *pointer = INFO.Fetch[DIVindex];
        }
}

template <bool loadfetch, bool writeback, typename scalar_t>
__device__ void fetchMatrix(
    MatrixInfo<scalar_t> &INFO,
    const ProblemInfo &INFO_P,
    const size_t &JOBindex,
    const size_t &Worker
) {
    scalar_t *offset = getBlockOffset<scalar_t>(INFO, INFO_P, JOBindex);
    scalar_t *pointer;
    if(INFO.prev_offset != offset)
    {
        if(writeback)
        {
            writebackMatrix<scalar_t>(INFO, INFO_P, Worker);
            INFO.last_Jobindex = JOBindex;
        }
        if(loadfetch)
            // TODO: P0, pointer is not global memory frendly
            for(size_t DIVindex=0; DIVindex<INFO.fetch_numel; DIVindex+=Worker)
            {
                pointer = getDivisionRelativeIndex<scalar_t>(
                    INFO, INFO_P, JOBindex, DIVindex, offset
                );
                if(pointer != nullptr)
                    INFO.Fetch[DIVindex] = *pointer;
                else
                    INFO.Fetch[DIVindex] = 0;
            }
        else
            for(size_t DIVindex=0; DIVindex<INFO.fetch_numel; DIVindex+=Worker)
                INFO.Fetch[DIVindex] = 0;
    }
    INFO.prev_offset = offset;
}

template <typename scalar_t>
__device__ void accFetch(
    MatrixInfo<scalar_t> &INFO_A,
    MatrixInfo<scalar_t> &INFO_B,
    MatrixInfo<scalar_t> &INFO_C,
    const ProblemInfo &INFO_P,
    const size_t &Worker
) {
    // TODO: P0, reduction axis do not share over thread
    for(size_t i=0; i<INFO_P.TaskSize; i+=Worker)
        INFO_C.Fetch[getFetchRelativeIndex<scalar_t>(INFO_C, INFO_P, i)] += (
            INFO_A.Fetch[getFetchRelativeIndex<scalar_t>(INFO_A, INFO_P, i)]
            * 
            INFO_B.Fetch[getFetchRelativeIndex<scalar_t>(INFO_B, INFO_P, i)]
        );
}

template <typename scalar_t>
__global__ void PlaneDot(
    MatrixInfo<scalar_t> INFO_A,
    MatrixInfo<scalar_t> INFO_B,
    MatrixInfo<scalar_t> INFO_C,
    ProblemInfo INFO_P
) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_uchar[];
    scalar_t *fetch = reinterpret_cast<scalar_t *>(sdata_uchar);
    size_t JOBindex=INFO_P.getJobIndex(blockIdx.x, gridDim.x);
    size_t JOBlast=INFO_P.getJobIndex(blockIdx.x+1, gridDim.x);
    scalar_t *fetch_pos = fetch;
    fetch_pos = INFO_A.initialize(fetch_pos, 0);
    fetch_pos = INFO_B.initialize(fetch_pos, INFO_A.fetch_numel);
    fetch_pos = INFO_C.initialize(fetch_pos, INFO_B.fetch_numel);

    for(; JOBindex < JOBlast; JOBindex++)
    {
        fetchMatrix<true, false, scalar_t>(INFO_A, INFO_P, JOBindex, blockDim.x);
        fetchMatrix<true, false, scalar_t>(INFO_B, INFO_P, JOBindex, blockDim.x);
        fetchMatrix<false, true, scalar_t>(INFO_C, INFO_P, JOBindex, blockDim.x);
        __syncthreads();
        accFetch<scalar_t>(INFO_A, INFO_B, INFO_C, INFO_P, blockDim.x);
        __syncthreads();
    }
    writebackMatrix<scalar_t>(INFO_C, INFO_P, blockDim.x);
}

void determin_task_divide(size_t works, size_t device, size_t&nBlock, size_t&nThread) {
    cudaDeviceProp *prop=at::cuda::getDeviceProperties(device);
    nBlock = min((size_t)prop->multiProcessorCount, works);
    nThread = prop->maxThreadsPerBlock;
    // printf("reduce: nblock: %d \t nThread: %d\t per thread: %.2f\n", nBlock, nThread, (float)works/nBlock/nThread);
}

void PlaneDot_wrapper(
    at::Tensor Mat_A,
    at::Tensor Mat_B,
    at::Tensor Mat_C,
    at::Tensor _DIM,             // [ndim]
    at::Tensor _DIV,             // [ndim]
    at::Tensor _ChunkDims,       // []
    at::Tensor _fetch_numel,     // [3]
    at::Tensor _effective_dim    // [3]
) {
    auto DIM = _DIM.accessor<int64_t, 1>();
    auto DIV = _DIV.accessor<int64_t, 1>();
    auto fetch_numel = _fetch_numel.accessor<int64_t, 1>();
    auto effective_dim = _effective_dim.accessor<int64_t, 1>();
    size_t nBlock, nThread;
    ProblemInfo INFO_P(DIM, DIV, _ChunkDims.item<int64_t>());
    determin_task_divide(INFO_P.DivisibleJobCount(), (size_t)Mat_A.device().index(), nBlock, nThread);
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, Mat_A.type(), "PlaneDot", ([&] {
        MatrixInfo<scalar_t> INFO_A(Mat_A, fetch_numel[0], effective_dim[0]);
        MatrixInfo<scalar_t> INFO_B(Mat_B, fetch_numel[1], effective_dim[1]);
        MatrixInfo<scalar_t> INFO_C(Mat_C, fetch_numel[2], effective_dim[2]);
        PlaneDot<scalar_t> <<<nBlock,nThread,_fetch_numel.sum().item<int64_t>()*sizeof(scalar_t),at::cuda::getCurrentCUDAStream()>>>(
            INFO_A, INFO_B, INFO_C, INFO_P
        );
    }));
}

int Get_MaxBlock(at::Tensor Mat_A) {
    return at::cuda::getDeviceProperties(Mat_A.device().index())->multiProcessorCount;
}
int Get_ShMem(at::Tensor Mat_A) {
    return at::cuda::getDeviceProperties(Mat_A.device().index())->sharedMemPerBlock;
}
int Get_WarpSize(at::Tensor Mat_A) {
    return at::cuda::getDeviceProperties(Mat_A.device().index())->warpSize;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("Get_MaxBlock", &Get_MaxBlock, "CUDA DeviceProp");
    m.def("Get_ShMem", &Get_ShMem, "CUDA DeviceProp");
    m.def("Get_WarpSize", &Get_WarpSize, "CUDA DeviceProp");
    m.def("PlaneDot", &PlaneDot_wrapper, "PlaneDot CUDA");
}
