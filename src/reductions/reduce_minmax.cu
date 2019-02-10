#include <duda/reductions/reduce_minmax.hpp>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <utility>

namespace duda
{

namespace detail
{

template <typename T>
inline std::pair<T, T> reduce_minmax(const T* const data, const int size)
{
    const auto ptr    = thrust::device_pointer_cast(data);
    const auto minmax = thrust::minmax_element(ptr, ptr + size);

    return {*minmax.first, *minmax.second};
}

} // namespace detail

std::pair<int, int> reduce_minmax(const int* const data, const int size)
{
    return detail::reduce_minmax(data, size);
}

std::pair<float, float> reduce_minmax(const float* const data, const int size)
{
    return detail::reduce_minmax(data, size);
}

std::pair<double, double> reduce_minmax(const double* const data, const int size)
{
    return detail::reduce_minmax(data, size);
}

} // namespace duda
