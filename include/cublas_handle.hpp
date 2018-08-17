#ifndef DUDA_CUBLAS_HANDLE_HPP_
#define DUDA_CUBLAS_HANDLE_HPP_

#include "check_error.hpp"

#include <cublas_v2.h>

namespace duda
{

struct cublas_handle_wrapper
{
    cublas_handle_wrapper()
    {
        check_cublas_error(cublasCreate(&handle_));
    }

    ~cublas_handle_wrapper()
    {
        check_cublas_error(cublasDestroy(handle_));
    }

    cublasHandle_t& value()
    {
        return handle_;
    }

private:
    cublasHandle_t handle_;
};

cublas_handle_wrapper& cublas_handle()
{
    static cublas_handle_wrapper handle;
    return handle;
}

} // namespace duda

#endif /* DUDA_CUBLAS_HANDLE_HPP_ */
