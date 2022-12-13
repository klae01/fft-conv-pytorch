#include <ATen/ATen.h>
#include <torch/extension.h>

#include <algorithm>
using namespace std;

template<typename T>
void loopbincount(
    vector<T> &data,
    vector<T> &result,
    const T &length,
    const T &&loop,
    const T &&remain
) {
    size_t i;
    T multiple = loop + 1;
    for(i=0; i<remain; i++)
        result[data[i]] += multiple;
    if(loop)
        for(i=remain; i<data.size(); i++)
            result[data[i]] += loop;
}
void inplace_convolve(
    const auto &signal,
    const auto &kernel,
    auto &output
) {
    size_t size=signal.size(), i, j, k;
    for(i=0; i<size; i++)
    {
        for(j=0, k=i; k<size; j++, k++)
            output[i] += signal[k] * kernel[j];
        for(j=size-i, k=0; j<size; j++, k++)
            output[i] += signal[k] * kernel[j];
    }
}
template<typename T>
inline void inplace_acc_mod(
    vector<T> &V,
    T init,
    T delta,
    const T &modulo
) {
    init %= modulo;
    delta %= modulo;
    T threshold = modulo - delta;
    for(auto &I:V) {
        I = init;
        if(init < threshold)
            init += delta;
        else init -= threshold;
    }
}
template<typename T>
inline T GCD(T x, T y)
{
    T t;
    while(y) {t=y; y=x%y; x=t;}
    return x;
}
template<typename T>
inline T LCM(T x, T y) { return x / GCD(x, y) * y; }
template<typename datatype, typename acctype>
acctype access_counter(
    const datatype ndim,
    const at::TensorAccessor<datatype, 1> DIM,
    const at::TensorAccessor<datatype, 1> DIV,
    const at::TensorAccessor<datatype, 1> stride,
    const datatype chunksize
) {
    vector<acctype> prev_access(chunksize, 1);
    vector<acctype> curr_access(chunksize);
    vector<datatype> prev_remain(chunksize);
    vector<datatype> curr_remain(chunksize);
    vector<datatype> index(chunksize);
    vector<datatype> bin(chunksize);
    vector<datatype> step_order(ndim);
    datatype tmp, i, lcm;
    bool keep_overlap=true;
    tmp=ndim; for(auto &I:step_order) I=--tmp;
    for(i=1; i<ndim && stride[i-1]>=stride[i]; i++);
    if(i<ndim)
        stable_sort(step_order.begin(), step_order.end(), [&v=stride](size_t i1, size_t i2) {return v[i1] < v[i2];});
    tmp=chunksize - stride[step_order[0]];
    inplace_acc_mod<datatype>(prev_remain, tmp, tmp, chunksize);
    for(auto &step:step_order)
    {
        fill(curr_access.begin(), curr_access.end(), 0);
        fill(bin.begin(), bin.end(), 0);
        index.resize(chunksize);
        inplace_acc_mod<datatype>(index, 0, stride[step], chunksize);
        loopbincount<datatype>(index, bin, chunksize, DIM[step] / chunksize, DIM[step] % chunksize);
        inplace_convolve(prev_access, bin, curr_access);
        
        if(keep_overlap)
        {
            // compute remains
            tmp=stride[step]*(DIM[step]-DIV[step]);
            keep_overlap=false;
            for(i=0;i<chunksize;i++)
                if(prev_remain[index[i]] > tmp)
                {
                    curr_remain[i] = prev_remain[index[i]] - tmp;
                    keep_overlap = true;
                }
                else curr_remain[i] = 0;

            fill(bin.begin(), bin.end(), 0);
            lcm = LCM(DIV[step], chunksize);
            // all case
            loopbincount<datatype>(index, bin, chunksize, (DIM[step] - 1) / chunksize, (DIM[step] - 1) % chunksize);
            for(auto &I:bin) I=-I;
            // non overlap case
            index.resize(lcm / DIV[step]);
            inplace_acc_mod<datatype>(index, (DIV[step] - 1) * stride[step], DIV[step] * stride[step], chunksize);
            loopbincount<datatype>(index, bin, chunksize, (DIM[step] - 1) / lcm, ((DIM[step] - 1) % lcm + 1) / DIV[step]);

            for(auto &I:prev_remain) I=(I>=stride[step]);
            inplace_convolve(prev_remain, bin, curr_access);
        }
        swap(prev_access, curr_access);
        swap(prev_remain, curr_remain);
    }
    return prev_access[0];
}
void access_count_wrapper(
    at::Tensor ndim, // shared, shape [1]
    at::Tensor chunksize, // shared, shape []
    at::Tensor DIM, // shared, shape [ndim]
    at::Tensor stride, // shared, shape [ndim]
    at::Tensor DIV, // non shared, shape [tasksize, ndim]
    at::Tensor result // non shared, shape [tasksize]
) {
    AT_DISPATCH_INTEGRAL_TYPES(DIM.type(), "Access Count", ([&] {
        scalar_t NDIM = ndim.accessor<scalar_t,1>()[0];
        scalar_t CHUNK = chunksize.item<scalar_t>();
        auto DIM_pt=DIM.accessor<scalar_t,1>();
        auto stride_pt=stride.accessor<scalar_t,1>();
        auto DIV_pt=DIV.accessor<scalar_t,2>();
        for(size_t i=0; i<result.sizes()[0]; i++)
            result[i] = access_counter<scalar_t, float> (
                NDIM, DIM_pt, DIV_pt[i], stride_pt, CHUNK
            );
    }));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("AccessCost", &access_count_wrapper, "Memory Access cost estimation");
}
