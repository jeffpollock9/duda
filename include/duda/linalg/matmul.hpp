#ifndef DUDA_LINALG_MATMUL_HPP_
#define DUDA_LINALG_MATMUL_HPP_

#include <duda/blas/level3.hpp>
#include <duda/device_matrix.hpp>
#include <duda/utility/dim.hpp>
#include <duda/utility/enums.hpp>

namespace duda
{

template <typename T>
device_matrix<T> matmul(const device_matrix<T>& a,
                        const device_matrix<T>& b,
                        const op op_a = op::none,
                        const op op_b = op::none,
                        const T alpha = 1.0)
{
    constexpr T beta = 0.0;

    const dim dim_op_a = dim(a, op_a);
    const dim dim_op_b = dim(b, op_b);

    device_matrix<T> c(dim_op_a.rows, dim_op_b.cols);

    duda::gemm(op_a, op_b, alpha, a, b, beta, c);

    return c;
}

} // namespace duda

#endif /* DUDA_LINALG_MATMUL_HPP_ */
