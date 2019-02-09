#ifndef DUDA_UTILITY_ENUMS_HPP_
#define DUDA_UTILITY_ENUMS_HPP_

#include <cublas_v2.h>

#include <type_traits>

namespace duda
{

enum class op : std::underlying_type_t<cublasOperation_t> {
    none                = CUBLAS_OP_N,
    transpose           = CUBLAS_OP_T,
    conjugate_transpose = CUBLAS_OP_C
};

enum class fill_mode : std::underlying_type_t<cublasFillMode_t> {
    lower = CUBLAS_FILL_MODE_LOWER,
    upper = CUBLAS_FILL_MODE_UPPER
};

} // namespace duda

#endif /* DUDA_UTILITY_ENUMS_HPP_ */
