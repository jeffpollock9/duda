
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
    const dim3 grid_dim(rows, cols);
    const dim3 block_dim(1, 1);

    eye_kernel<double><<<grid_dim, block_dim>>>(data, rows, cols);
}

} // namespace duda
