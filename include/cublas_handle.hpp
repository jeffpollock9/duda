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
        const auto code = cublasCreate(&handle_);

        check_cublas_error(code);
    }

    ~cublas_handle_wrapper()
    {
        const auto code = cublasDestroy(handle_);

        check_cublas_error(code);
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
