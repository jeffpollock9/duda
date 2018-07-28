#ifndef DUDA_CHECK_ERROR_HPP_
#define DUDA_CHECK_ERROR_HPP_

#include "macros.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <curand.h>

#include <stdexcept>
#include <string>

namespace duda
{

inline void check_cuda_error(const cudaError_t code)
{
    if (DUDA_UNLIKELY(code != cudaSuccess))
    {
        throw std::runtime_error("cuda error code: " + std::to_string(code));
    }
}

inline void check_cublas_error(const cublasStatus_t code)
{
    if (DUDA_UNLIKELY(code != CUBLAS_STATUS_SUCCESS))
    {
        throw std::runtime_error("cublas error code: " + std::to_string(code));
    }
}

inline void check_curand_error(const curandStatus_t code)
{
    if (DUDA_UNLIKELY(code != CURAND_STATUS_SUCCESS))
    {
        throw std::runtime_error("curand error code: " + std::to_string(code));
    }
}

} // namespace duda

#endif /* DUDA_CHECK_ERROR_HPP_ */
