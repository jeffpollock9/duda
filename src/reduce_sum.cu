#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

namespace duda
{

namespace detail
{

template <typename T>
inline T reduce_sum(const T* const data, const int size)
{
    T* out_d          = NULL;
    void* tmp_storage = NULL;

    const auto code1 = cudaMalloc((void**)&out_d, sizeof(T));

    std::size_t tmp_storage_bytes = 0;

    const auto code2 = cub::DeviceReduce::Sum(
        tmp_storage, tmp_storage_bytes, data, out_d, size);

    const auto code3 = cudaMalloc(&tmp_storage, tmp_storage_bytes);

    const auto code4 = cub::DeviceReduce::Sum(
        tmp_storage, tmp_storage_bytes, data, out_d, size);

    T* out_h = new T[1];

    cudaMemcpy(out_h, out_d, sizeof(T) * 1, cudaMemcpyDeviceToHost);

    const T out = out_h[0];

    const auto code5 = cudaFree(tmp_storage);

    const auto code6 = cudaFree(out_d);

    delete[] out_h;

    return out;
}

} // namespace detail

float reduce_sum(const float* const data, const int size)
{
    return detail::reduce_sum(data, size);
}

double reduce_sum(const double* const data, const int size)
{
    return detail::reduce_sum(data, size);
}

} // namespace duda
