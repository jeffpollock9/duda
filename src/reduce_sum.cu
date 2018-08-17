#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

namespace duda
{

namespace detail
{

template <typename T>
inline T reduce_sum(T* const data, const int size)
{
    // storage for answer
    T* out_d = NULL;

    const cudaError_t code1 = cudaMalloc((void**)&out_d, sizeof(T) * 1);

    // get temp storage
    void* tmp_storage = NULL;

    std::size_t tmp_storage_bytes = 0;

    const cudaError_t code2 = cub::DeviceReduce::Sum(
        tmp_storage, tmp_storage_bytes, data, out_d, size);

    const cudaError_t code3 = cudaMalloc(&tmp_storage, tmp_storage_bytes);

    // do the reduction
    const cudaError_t code4 = cub::DeviceReduce::Sum(
        tmp_storage, tmp_storage_bytes, data, out_d, size);

    // copy answer to host
    T* out_h = new T[1];

    cudaMemcpy(out_h, out_d, sizeof(T) * 1, cudaMemcpyDeviceToHost);

    const T out = out_h[0];

    // tidy up
    const cudaError_t code5 = cudaFree(tmp_storage);

    const cudaError_t code6 = cudaFree(out_d);

    delete[] out_h;

    return out;
}

} // namespace detail

float reduce_sum(float* const data, const int size)
{
    return detail::reduce_sum(data, size);
}

double reduce_sum(double* const data, const int size)
{
    return detail::reduce_sum(data, size);
}

} // namespace duda
