#ifndef DUDA_UTILITY_CHECK_ERROR_HPP_
#define DUDA_UTILITY_CHECK_ERROR_HPP_

#include <duda/utility/macros.hpp>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <rmm/rmm.h>

#include <stdexcept>
#include <string>

namespace duda
{

inline void check_error(const rmmError_t code)
{
    if (DUDA_UNLIKELY(code != rmmError_t::RMM_SUCCESS))
    {
        throw std::runtime_error(std::string("rmm error code: ") +
                                 rmmGetErrorString(code));
    }
}

inline void check_error(const cudaError_t code)
{
    if (DUDA_UNLIKELY(code != cudaError_t::cudaSuccess))
    {
        throw std::runtime_error(std::string("cuda error: ") +
                                 cudaGetErrorString(code));
    }
}

inline void check_error(const cublasStatus_t code)
{
    if (DUDA_UNLIKELY(code != CUBLAS_STATUS_SUCCESS))
    {
        throw std::runtime_error("cublas error code: " + std::to_string(code));
    }
}

inline void check_error(const curandStatus_t code)
{
    if (DUDA_UNLIKELY(code != CURAND_STATUS_SUCCESS))
    {
        throw std::runtime_error("curand error code: " + std::to_string(code));
    }
}

} // namespace duda

#endif /* DUDA_UTILITY_CHECK_ERROR_HPP_ */
