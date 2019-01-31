#ifndef DUDA_BLAS_LEVEL2_HPP_
#define DUDA_BLAS_LEVEL2_HPP_

#include <duda/cublas_handle.hpp>
#include <duda/detail.hpp>
#include <duda/device_matrix.hpp>
#include <duda/device_vector.hpp>
#include <duda/dim.hpp>
#include <duda/macros.hpp>
#include <duda/op.hpp>

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

} // namespace duda

#endif /* DUDA_BLAS_LEVEL2_HPP_ */
