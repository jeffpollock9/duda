#ifndef DUDA_BLAS_HPP_
#define DUDA_BLAS_HPP_

#include <cublas_v2.h>

#include "cublas_handle.hpp"
#include "detail.hpp"
#include "device_matrix.hpp"

namespace duda
{

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
inline void gemm(const T alpha,
                 const device_matrix<T>& A,
                 const device_matrix<T>& B,
                 const T beta,
                 device_matrix<T>& C)
{
    const int m = A.rows();
    const int n = B.cols();
    const int k = A.cols();

    const int lda = A.rows();
    const int ldb = B.rows();
    const int ldc = C.rows();

    const auto code = detail::overload<T>::call(cublasSgemm,
                                                cublasDgemm,
                                                cublas_handle().value(),
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                m,
                                                n,
                                                k,
                                                &alpha,
                                                A.data(),
                                                lda,
                                                B.data(),
                                                ldb,
                                                &beta,
                                                C.data(),
                                                ldc);

    check_cublas_error(code);
}

} // namespace duda

#endif /* DUDA_BLAS_HPP_ */
