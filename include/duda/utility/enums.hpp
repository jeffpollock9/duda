#ifndef DUDA_UTILITY_ENUMS_HPP_
#define DUDA_UTILITY_ENUMS_HPP_

#include <cublas_v2.h>
#include <rmm/rmm.h>

#include <type_traits>

namespace duda
{

enum class op : std::underlying_type_t<cublasOperation_t> {
    none                = cublasOperation_t::CUBLAS_OP_N,
    transpose           = cublasOperation_t::CUBLAS_OP_T,
    conjugate_transpose = cublasOperation_t::CUBLAS_OP_C
};

enum class fill_mode : std::underlying_type_t<cublasFillMode_t> {
    lower = cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
    upper = cublasFillMode_t::CUBLAS_FILL_MODE_UPPER
};

enum class allocation_mode : std::underlying_type_t<rmmAllocationMode_t> {
    cuda_default_allocation = rmmAllocationMode_t::CudaDefaultAllocation,
    pool_allocation         = rmmAllocationMode_t::PoolAllocation,
    cuda_managed_memory     = rmmAllocationMode_t::CudaManagedMemory
};

} // namespace duda

#endif /* DUDA_UTILITY_ENUMS_HPP_ */
