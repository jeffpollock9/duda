#include <duda/math/log.hpp>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace duda
{

namespace detail
{

template <typename T>
struct log_functor
{
    __device__ T operator()(const T x) const
    {
        return std::log(x);
    }
};

template <typename T>
inline void log(T* const data, const int size)
{
    const auto ptr = thrust::device_pointer_cast(data);

    thrust::transform(ptr, ptr + size, ptr, log_functor<T>{});
}

} // namespace detail

void log(float* const data, const int size)
{
    detail::log(data, size);
}

void log(double* const data, const int size)
{
    detail::log(data, size);
}

} // namespace duda
