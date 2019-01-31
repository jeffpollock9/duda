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
inline void amax(const device_vector<T>& x, int& result)
{
    const int incx = 1;

    const auto code = detail::overload<T>::call(cublasIsamax,
                                                cublasIdamax,
                                                cublas_handle().value(),
                                                x.size(),
                                                x.data(),
                                                incx,
                                                &result);

    check_error(code);

    // use 0-based indexing
    --result;
}

template <typename T>
inline void amin(const device_vector<T>& x, int& result)
{
    const int incx = 1;

    const auto code = detail::overload<T>::call(cublasIsamin,
                                                cublasIdamin,
                                                cublas_handle().value(),
                                                x.size(),
                                                x.data(),
                                                incx,
                                                &result);

    check_error(code);

    // use 0-based indexing
    --result;
}

template <template <typename> class Device, typename T>
inline void asum(const Device<T>& x, T& result)
{
    const int incx = 1;

    const auto code = detail::overload<T>::call(cublasSasum,
                                                cublasDasum,
                                                cublas_handle().value(),
                                                x.size(),
                                                x.data(),
                                                incx,
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

    const int incx = 1;
    const int incy = 1;

    const auto code = detail::overload<T>::call(cublasSaxpy,
                                                cublasDaxpy,
                                                cublas_handle().value(),
                                                x.size(),
                                                &alpha,
                                                x.data(),
                                                incx,
                                                y.data(),
                                                incy);

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

    const int incx = 1;
    const int incy = 1;

    const auto code = detail::overload<T>::call(cublasSdot,
                                                cublasDdot,
                                                cublas_handle().value(),
                                                n,
                                                x.data(),
                                                incx,
                                                y.data(),
                                                incy,
                                                &result);

    check_error(code);
}

} // namespace duda

#endif /* DUDA_BLAS_LEVEL1_HPP_ */
