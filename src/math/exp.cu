#include <duda/math/exp.hpp>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace duda
{

namespace detail
{

template <typename T>
struct exp_functor
{
    __device__ T operator()(const T x) const
    {
        return std::exp(x);
    }
};

template <typename T>
inline void exp(T* const data, const int size)
{
    const auto ptr = thrust::device_pointer_cast(data);

    thrust::transform(ptr, ptr + size, ptr, exp_functor<T>{});
}

} // namespace detail

void exp(float* const data, const int size)
{
    detail::exp(data, size);
}

void exp(double* const data, const int size)
{
    detail::exp(data, size);
}

} // namespace duda
