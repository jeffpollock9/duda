#ifndef DUDA_BLAS_LEVEL1_HPP_
#define DUDA_BLAS_LEVEL1_HPP_

#include <duda/cublas_handle.hpp>
#include <duda/detail.hpp>
#include <duda/device_vector.hpp>
#include <duda/dim.hpp>
#include <duda/macros.hpp>

#include <cublas_v2.h>

namespace duda
{

template <typename T>
inline void iamax(const device_vector<T>& x, int& result)
{
    const auto code = detail::overload<T>::call(cublasIsamax,
                                                cublasIdamax,
                                                cublasIcamax,
                                                cublasIzamax,
                                                cublas_handle().value(),
                                                x.size(),
                                                x.data(),
                                                detail::incx(),
                                                &result);

    check_error(code);

    // use 0-based indexing
    --result;
}

template <typename T>
inline void iamin(const device_vector<T>& x, int& result)
{
    const auto code = detail::overload<T>::call(cublasIsamin,
                                                cublasIdamin,
                                                cublasIcamin,
                                                cublasIzamin,
                                                cublas_handle().value(),
                                                x.size(),
                                                x.data(),
                                                detail::incx(),
                                                &result);

    check_error(code);

    // use 0-based indexing
    --result;
}

template <template <typename> class Device, typename T>
inline void asum(const Device<T>& x, T& result)
{
    const auto code = detail::overload<T>::call(cublasSasum,
                                                cublasDasum,
                                                cublasScasum,
                                                cublasDzasum,
                                                cublas_handle().value(),
                                                x.size(),
                                                x.data(),
                                                detail::incx(),
                                                &result);

    check_error(code);
}

template <template <typename> class Device, typename T>
inline void axpy(const T alpha, const Device<T>& x, Device<T>& y)
{
    const dim dim_x(x);
    const dim dim_y(y);

    if (DUDA_UNLIKELY(dim_x != dim_y))
    {
        throw std::runtime_error("can't axpy with dimensions " + dim_x +
                                 " and " + dim_y);
    }

    const auto code = detail::overload<T>::call(cublasSaxpy,
                                                cublasDaxpy,
                                                cublasCaxpy,
                                                cublasZaxpy,
                                                cublas_handle().value(),
                                                x.size(),
                                                &alpha,
                                                x.data(),
                                                detail::incx(),
                                                y.data(),
                                                detail::incy());

    check_error(code);
}

template <typename T>
inline void dot(const device_vector<T>& x, const device_vector<T>& y, T& result)
{
    const int n = x.size();

    if (DUDA_UNLIKELY(n != y.size()))
    {
        using std::to_string;

        throw std::runtime_error("can't dot with sizes " + to_string(n) +
                                 " and " + to_string(y.size()));
    }

    const auto code = detail::overload<T>::call(cublasSdot,
                                                cublasDdot,
                                                detail::not_callable{},
                                                detail::not_callable{},
                                                cublas_handle().value(),
                                                n,
                                                x.data(),
                                                detail::incx(),
                                                y.data(),
                                                detail::incy(),
                                                &result);

    check_error(code);
}

template <typename T>
inline void nrm2(const device_vector<T>& x, T& result)
{
    const int n = x.size();

    const auto code = detail::overload<T>::call(cublasSnrm2,
                                                cublasDnrm2,
                                                cublasScnrm2,
                                                cublasDznrm2,
                                                cublas_handle().value(),
                                                n,
                                                x.data(),
                                                detail::incx(),
                                                &result);

    check_error(code);
}

} // namespace duda

#endif /* DUDA_BLAS_LEVEL1_HPP_ */
