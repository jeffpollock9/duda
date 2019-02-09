#ifndef DUDA_BLAS_LEVEL2_HPP_
#define DUDA_BLAS_LEVEL2_HPP_

#include <duda/blas/cublas_handle.hpp>
#include <duda/detail/overload.hpp>
#include <duda/detail/inc.hpp>
#include <duda/device_matrix.hpp>
#include <duda/device_vector.hpp>
#include <duda/utility/dim.hpp>
#include <duda/utility/enums.hpp>
#include <duda/utility/macros.hpp>

#include <cublas_v2.h>

namespace duda
{

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

    const auto fn = detail::overload<T>::fn(
        cublasSgemv, cublasDgemv, cublasCgemv, cublasZgemv);

    const auto code = fn(cublas_handle().value(),
                         static_cast<cublasOperation_t>(op_a),
                         m,
                         n,
                         &alpha,
                         a.data(),
                         lda,
                         x.data(),
                         detail::incx(),
                         &beta,
                         y.data(),
                         detail::incy());

    check_error(code);
}

template <typename T>
inline void syr(const fill_mode uplo,
                const T alpha,
                const device_vector<T>& x,
                device_matrix<T>& a)
{
    const int n = x.size();

    if (DUDA_UNLIKELY(n != a.rows() || n != a.cols()))
    {
        throw std::runtime_error("can't syr with input size " +
                                 std::to_string(n) +
                                 " and output dimension + " + dim(a));
    }

    const int lda = a.rows();

    const auto fn =
        detail::overload<T>::fn(cublasSsyr, cublasDsyr, cublasCsyr, cublasZsyr);

    const auto code = fn(cublas_handle().value(),
                         static_cast<cublasFillMode_t>(uplo),
                         n,
                         &alpha,
                         x.data(),
                         detail::incx(),
                         a.data(),
                         lda);

    check_error(code);
}

} // namespace duda

#endif /* DUDA_BLAS_LEVEL2_HPP_ */
