#include <duda/fill.hpp>

namespace duda
{

namespace detail
{

template <typename T>
__global__ void fill_kernel(T* const data, const int size, const T value)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
    {
        data[i] = value;
    }
}

template <typename T>
inline void fill(T* const data, const int size, const T value)
{
    const int d = 1024;
    const int n = (size + d) / d;

    const dim3 blocks(n);
    const dim3 block_dim(d);

    fill_kernel<<<blocks, block_dim>>>(data, size, value);
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
