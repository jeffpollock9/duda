#ifndef DEVICE_MATRIX_HPP_
#define DEVICE_MATRIX_HPP_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "check_error.hpp"
#include "cublas_handle.hpp"
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

    static device_matrix random(const int rows, const int cols)
    {
        device_matrix x(rows, cols);

        fill_random_uniform(x.data(), x.size());

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

inline void axpy(const double alpha,
                 const device_matrix<double>& x,
                 device_matrix<double>& y)
{
    const auto code = cublasDaxpy(
        cublas_handle().value(), x.size(), &alpha, x.data(), 1, y.data(), 1);

    check_cublas_error(code);
}

inline void
axpy(const float alpha, const device_matrix<float>& x, device_matrix<float>& y)
{
    const auto code = cublasSaxpy(
        cublas_handle().value(), x.size(), &alpha, x.data(), 1, y.data(), 1);

    check_cublas_error(code);
}

inline void gemm(const double alpha,
                 const device_matrix<double>& A,
                 const device_matrix<double>& B,
                 const double beta,
                 device_matrix<double>& C)
{
    const int m = A.rows();
    const int n = B.cols();
    const int k = A.cols();

    const int ld = 1;

    const auto code = cublasDgemm(cublas_handle().value(),
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  m,
                                  n,
                                  k,
                                  &alpha,
                                  A.data(),
                                  ld,
                                  B.data(),
                                  ld,
                                  &beta,
                                  C.data(),
                                  ld);

    check_cublas_error(code);
}

inline void gemm(const float alpha,
                 const device_matrix<float>& A,
                 const device_matrix<float>& B,
                 const float beta,
                 device_matrix<float>& C)
{
    const int m = A.rows();
    const int n = B.cols();
    const int k = A.cols();

    const int ld = 1;

    const auto code = cublasSgemm(cublas_handle().value(),
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  m,
                                  n,
                                  k,
                                  &alpha,
                                  A.data(),
                                  ld,
                                  B.data(),
                                  ld,
                                  &beta,
                                  C.data(),
                                  ld);

    check_cublas_error(code);
}

} // namespace duda

#endif /* DEVICE_MATRIX_HPP_ */
