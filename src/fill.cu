#include <duda/fill.hpp>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace duda
{

namespace detail
{

template <typename T>
void fill(T* const data, const int size, const T value)
{
    const auto ptr = thrust::device_pointer_cast(data);
    thrust::fill(ptr, ptr + size, value);
}

} // namespace detail

void fill(int* const data, const int size, const int value)
{
    detail::fill(data, size, value);
}

void fill(float* const data, const int size, const float value)
{
    detail::fill(data, size, value);
}

void fill(double* const data, const int size, const double value)
{
    detail::fill(data, size, value);
}

} // namespace duda
