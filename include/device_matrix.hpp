#ifndef DEVICE_MATRIX_HPP_
#define DEVICE_MATRIX_HPP_

#include <cuda_runtime_api.h>

#include "check_error.hpp"
#include "curand_generator.hpp"
#include "random.hpp"

namespace duda
{

template <typename T>
struct device_matrix
{
    device_matrix() = default;

    device_matrix(const int rows, const int cols) : rows_(rows), cols_(cols)
    {
        const auto code = cudaMalloc((void**)&data_, bytes());

        check_cuda_error(code);
    }

    device_matrix(const T* const host, const int rows, const int cols)
        : device_matrix(rows, cols)
    {
        const auto code =
            cudaMemcpy(data(), host, bytes(), cudaMemcpyHostToDevice);

        check_cuda_error(code);
    }

    ~device_matrix()
    {
        const auto code = cudaFree(data());

        check_cuda_error(code);
    }

    static device_matrix random_uniform(const int rows, const int cols)
    {
        device_matrix x(rows, cols);

        fill_random_uniform(x.data(), x.size());

        return x;
    }

    static device_matrix random_normal(const int rows,
                                       const int cols,
                                       const T mean   = 0.0,
                                       const T stddev = 1.0)
    {
        device_matrix x(rows, cols);

        fill_random_normal(x.data(), x.size(), mean, stddev);

        return x;
    }

    int rows() const
    {
        return rows_;
    }

    int cols() const
    {
        return cols_;
    }

    int size() const
    {
        return rows() * cols();
    }

    int bytes() const
    {
        return size() * sizeof(T);
    }

    T* data()
    {
        return data_;
    }

    const T* data() const
    {
        return data_;
    }

private:
    T* data_  = nullptr;
    int rows_ = 0;
    int cols_ = 0;
};

template <typename T>
inline void copy(const device_matrix<T>& device, T* const host)
{
    const auto code =
        cudaMemcpy(host, device.data(), device.bytes(), cudaMemcpyDeviceToHost);

    check_cuda_error(code);
}

} // namespace duda

#endif /* DEVICE_MATRIX_HPP_ */
