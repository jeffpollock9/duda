#include <duda/reductions/reduce_sum.hpp>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

namespace duda
{

namespace detail
{

template <typename T>
inline T reduce_sum(const T* const data, const int size)
{
    const auto ptr = thrust::device_pointer_cast(data);
    return thrust::reduce(ptr, ptr + size);
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
