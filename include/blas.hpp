#ifndef DUDA_BLAS_HPP_
#define DUDA_BLAS_HPP_

#include "cublas_handle.hpp"
#include "detail.hpp"
#include "device_matrix.hpp"
#include "dim.hpp"
#include "macros.hpp"
#include "op.hpp"

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

namespace duda
{

template <template <typename> class Device, typename T>
inline void axpy(const T alpha, const Device<T>& x, Device<T>& y)
{
    const dim dim_x(x);
    const dim dim_y(y);

    if (DUDA_UNLIKELY(dim_x != dim_y))
    {
        throw std::runtime_error("can't axpy with dimensions " + dim_x +
                                 " and " + dim_y);
    }

    const int incx = 1;
    const int incy = 1;

    const auto code = detail::overload<T>::call(cublasSaxpy,
                                                cublasDaxpy,
                                                cublas_handle().value(),
                                                x.size(),
                                                &alpha,
                                                x.data(),
                                                incx,
                                                y.data(),
                                                incy);

    check_error(code);
}

template <typename T>
inline void gemm(const op op_a,
                 const op op_b,
                 const T alpha,
                 const device_matrix<T>& a,
                 const device_matrix<T>& b,
                 const T beta,
                 device_matrix<T>& c)
{
    const dim dim_op_a = dim(a, op_a);
    const dim dim_op_b = dim(b, op_b);
    const dim dim_c    = dim(c);

    if (DUDA_UNLIKELY(dim_op_a.cols != dim_op_b.rows))
    {
        throw std::runtime_error("can't gemm with input dimensions " +
                                 dim_op_a + " and " + dim_op_b);
    }

    if (DUDA_UNLIKELY(dim(dim_op_a.rows, dim_op_b.cols) != dim_c))
    {
        throw std::runtime_error("can't gemm with input dimensions " +
                                 dim_op_a + " and " + dim_op_b +
                                 " and output dimension " + dim_c);
    }

    const int m = dim_op_a.rows;
    const int n = dim_op_b.cols;
    const int k = dim_op_a.cols;

    const int lda = a.rows();
    const int ldb = b.rows();
    const int ldc = c.rows();

    const auto code =
        detail::overload<T>::call(cublasSgemm,
                                  cublasDgemm,
                                  cublas_handle().value(),
                                  static_cast<cublasOperation_t>(op_a),
                                  static_cast<cublasOperation_t>(op_b),
                                  m,
                                  n,
                                  k,
                                  &alpha,
                                  a.data(),
                                  lda,
                                  b.data(),
                                  ldb,
                                  &beta,
                                  c.data(),
                                  ldc);

    check_error(code);
}

template <typename T>
inline void gemv(const op op_a,
                 const T alpha,
                 const device_matrix<T>& a,
                 const device_vector<T>& x,
                 const T beta,
                 device_vector<T>& y)
{
    const dim dim_op_a = dim(a, op_a);

    if (DUDA_UNLIKELY(dim_op_a.cols != x.size()))
    {
        using std::to_string;

        throw std::runtime_error("can't gemv with input dimensions " +
                                 dim_op_a + " and " + to_string(x.size()));
    }

    if (DUDA_UNLIKELY(dim_op_a.rows != y.size()))
    {
        using std::to_string;

        throw std::runtime_error("can't gemv with input dimensions " +
                                 dim_op_a + " and " + to_string(x.size()) +
                                 " and output dimension " +
                                 to_string(y.size()));
    }

    const int m = a.rows();
    const int n = a.cols();

    const int lda = a.rows();

    const int incx = 1;
    const int incy = 1;

    const auto code =
        detail::overload<T>::call(cublasSgemv,
                                  cublasDgemv,
                                  cublas_handle().value(),
                                  static_cast<cublasOperation_t>(op_a),
                                  m,
                                  n,
                                  &alpha,
                                  a.data(),
                                  lda,
                                  x.data(),
                                  incx,
                                  &beta,
                                  y.data(),
                                  incy);

    check_error(code);
}

template <typename T>
inline void dot(const device_vector<T>& x, const device_vector<T>& y, T& result)
{
    const int n = x.size();

    if (DUDA_UNLIKELY(n != y.size()))
    {
        using std::to_string;

        throw std::runtime_error("can't dot with sizes " + to_string(n) +
                                 " and " + to_string(y.size()));
    }

    const int incx = 1;
    const int incy = 1;

    const auto code = detail::overload<T>::call(cublasSdot,
                                                cublasDdot,
                                                cublas_handle().value(),
                                                n,
                                                x.data(),
                                                incx,
                                                y.data(),
                                                incy,
                                                &result);

    check_error(code);
}

} // namespace duda

#endif /* DUDA_BLAS_HPP_ */
