#ifndef DUDA_OP_HPP_
#define DUDA_OP_HPP_

#include <cublas_v2.h>

#include <type_traits>

namespace duda
{

enum class op : std::underlying_type_t<cublasOperation_t> {
    none                = CUBLAS_OP_N,
    transpose           = CUBLAS_OP_T,
    conjugate_transpose = CUBLAS_OP_C
};

} // namespace duda

#endif /* DUDA_OP_HPP_ */
