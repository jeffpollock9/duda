#include <duda/reduce_sum.hpp>
#include <duda/check_error.hpp>

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

    T* out_d          = nullptr;
    void* tmp_storage = nullptr;

    check_error(cudaMalloc((void**)&out_d, sizeof(T)));

    std::size_t tmp_storage_bytes = 0;

    check_error(dr::Sum(tmp_storage, tmp_storage_bytes, data, out_d, size));

    check_error(cudaMalloc(&tmp_storage, tmp_storage_bytes));

    check_error(dr::Sum(tmp_storage, tmp_storage_bytes, data, out_d, size));

    check_error(cudaFree(tmp_storage));

    T out;

    check_error(cudaMemcpy(&out, out_d, sizeof(T), cudaMemcpyDeviceToHost));

    check_error(cudaFree(out_d));

    return out;
}

} // namespace detail

int reduce_sum(const int* const data, const int size)
{
    return detail::reduce_sum(data, size);
}

float reduce_sum(const float* const data, const int size)
{
    return detail::reduce_sum(data, size);
}

double reduce_sum(const double* const data, const int size)
{
    return detail::reduce_sum(data, size);
}

} // namespace duda
