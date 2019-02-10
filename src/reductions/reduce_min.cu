#include <duda/reductions/reduce_min.hpp>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

namespace duda
{

namespace detail
{

template <typename T>
inline T reduce_min(const T* const data, const int size)
{
    const auto ptr = thrust::device_pointer_cast(data);
    const auto min = thrust::min_element(ptr, ptr + size);

    return *min;
}

} // namespace detail

int reduce_min(const int* const data, const int size)
{
    return detail::reduce_min(data, size);
}

float reduce_min(const float* const data, const int size)
{
    return detail::reduce_min(data, size);
}

double reduce_min(const double* const data, const int size)
{
    return detail::reduce_min(data, size);
}

} // namespace duda
