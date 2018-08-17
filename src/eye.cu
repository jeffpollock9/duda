#include "eye.hpp"

namespace duda
{

namespace kernel
{

template <typename T>
__global__ void eye(T* const data, const int dim)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < dim && j < dim)
    {
        const int ix = i + j * dim;

        if (i == j)
        {
            data[ix] = 1;
        }
        else
        {
            data[ix] = 0;
        }
    }
}

} // namespace kernel

namespace detail
{

template <typename T>
inline void eye(T* const data, const int dim)
{
    const int d = 32;
    const int n = (dim + d) / d;

    const dim3 blocks(n, n);
    const dim3 block_dim(d, d);

    kernel::eye<T><<<blocks, block_dim>>>(data, dim);
}

} // namespace detail

void eye(float* const data, const int dim)
{
    detail::eye(data, dim);
}

void eye(double* const data, const int dim)
{
    detail::eye(data, dim);
}

} // namespace duda
