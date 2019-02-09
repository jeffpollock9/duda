#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

namespace duda
{

namespace detail
{

template <typename T>
inline T reduce_max(const T* const data, const int size)
{
    const auto ptr = thrust::device_pointer_cast(data);
    const auto max = thrust::max_element(ptr, ptr + size);

    return *max;
}

} // namespace detail

auto reduce_max(const int* const data, const int size)
{
    return detail::reduce_max(data, size);
}

auto reduce_max(const float* const data, const int size)
{
    return detail::reduce_max(data, size);
}

auto reduce_max(const double* const data, const int size)
{
    return detail::reduce_max(data, size);
}

} // namespace duda
