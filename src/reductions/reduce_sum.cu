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

auto reduce_sum(const int* const data, const int size)
{
    return detail::reduce_sum(data, size);
}

auto reduce_sum(const float* const data, const int size)
{
    return detail::reduce_sum(data, size);
}

auto reduce_sum(const double* const data, const int size)
{
    return detail::reduce_sum(data, size);
}

} // namespace duda
