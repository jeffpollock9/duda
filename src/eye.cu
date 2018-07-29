
namespace duda
{

template <typename T>
__global__ void eye_kernel(T* const data, const int rows, const int cols)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < rows && j < cols)
    {
        const int ix = i + j * rows;

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

void eye(double* const data, const int rows, const int cols)
{
    const dim3 blocks(1, 1);
    const dim3 threads_per_block(rows, cols);

    eye_kernel<double><<<blocks, threads_per_block>>>(data, rows, cols);
}

} // namespace duda
