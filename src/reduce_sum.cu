#include "reduce_sum.hpp"

#include "check_error.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

namespace duda
{

namespace detail
{

template <typename T>
inline T reduce_sum(const T* const data, const int size)
{
    using dr = cub::DeviceReduce;

    T* out_d          = NULL;
    void* tmp_storage = NULL;

    check_cuda_error(cudaMalloc((void**)&out_d, sizeof(T)));

    std::size_t tmp_storage_bytes = 0;

    check_cuda_error(
        dr::Sum(tmp_storage, tmp_storage_bytes, data, out_d, size));

    check_cuda_error(cudaMalloc(&tmp_storage, tmp_storage_bytes));

    check_cuda_error(
        dr::Sum(tmp_storage, tmp_storage_bytes, data, out_d, size));

    check_cuda_error(cudaFree(tmp_storage));

    T out;

    check_cuda_error(
        cudaMemcpy(&out, out_d, sizeof(T), cudaMemcpyDeviceToHost));

    check_cuda_error(cudaFree(out_d));

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
