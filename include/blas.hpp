#ifndef DUDA_BLAS_HPP_
#define DUDA_BLAS_HPP_

#include "cublas_handle.hpp"
#include "detail.hpp"
#include "device_matrix.hpp"

#include <cublas_v2.h>

#include <type_traits>

namespace duda
{

enum class op : std::underlying_type_t<cublasOperation_t> {
    none      = CUBLAS_OP_N,
    transpose = CUBLAS_OP_T
};

template <typename T>
inline void axpy(const T alpha, const device_matrix<T>& x, device_matrix<T>& y)
{
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

    check_cublas_error(code);
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
    const int m = a.rows();
    const int n = b.cols();
    const int k = a.cols();

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

    check_cublas_error(code);
}

} // namespace duda

#endif /* DUDA_BLAS_HPP_ */
