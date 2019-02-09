#ifndef DUDA_BLAS_LEVEL1_HPP_
#define DUDA_BLAS_LEVEL1_HPP_

#include <duda/blas/cublas_handle.hpp>
#include <duda/detail/inc.hpp>
#include <duda/detail/overload.hpp>
#include <duda/device_vector.hpp>
#include <duda/utility/dim.hpp>
#include <duda/utility/macros.hpp>

#include <cublas_v2.h>

namespace duda
{

template <typename T>
inline void iamax(const device_vector<T>& x, int& result)
{
    const auto fn = detail::overload<T>::fn(
        cublasIsamax, cublasIdamax, cublasIcamax, cublasIzamax);

    const auto code = fn(
        cublas_handle().value(), x.size(), x.data(), detail::incx(), &result);

    check_error(code);

    // use 0-based indexing
    --result;
}

template <typename T>
inline void iamin(const device_vector<T>& x, int& result)
{
    const auto fn = detail::overload<T>::fn(
        cublasIsamin, cublasIdamin, cublasIcamin, cublasIzamin);

    const auto code = fn(
        cublas_handle().value(), x.size(), x.data(), detail::incx(), &result);

    check_error(code);

    // use 0-based indexing
    --result;
}

template <template <typename> class DeviceStorage, typename T>
inline void asum(const DeviceStorage<T>& x, T& result)
{
    const auto fn = detail::overload<T>::fn(
        cublasSasum, cublasDasum, cublasScasum, cublasDzasum);

    const auto code = fn(
        cublas_handle().value(), x.size(), x.data(), detail::incx(), &result);

    check_error(code);
}

template <template <typename> class DeviceStorage, typename T>
inline void axpy(const T alpha, const DeviceStorage<T>& x, DeviceStorage<T>& y)
{
    const dim dim_x(x);
    const dim dim_y(y);

    if (DUDA_UNLIKELY(dim_x != dim_y))
    {
        throw std::runtime_error("can't axpy with dimensions " + dim_x +
                                 " and " + dim_y);
    }

    const auto fn = detail::overload<T>::fn(
        cublasSaxpy, cublasDaxpy, cublasCaxpy, cublasZaxpy);

    const auto code = fn(cublas_handle().value(),
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

    const auto fn = detail::overload<T>::fn(
        cublasSdot, cublasDdot, detail::not_callable{}, detail::not_callable{});

    const auto code = fn(cublas_handle().value(),
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
    const auto fn = detail::overload<T>::fn(
        cublasSnrm2, cublasDnrm2, cublasScnrm2, cublasDznrm2);

    const auto code = fn(
        cublas_handle().value(), x.size(), x.data(), detail::incx(), &result);

    check_error(code);
}

template <typename T>
inline void rot(device_vector<T>& x, device_vector<T>& y, const T c, const T s)
{
    const int n = x.size();

    if (DUDA_UNLIKELY(n != y.size()))
    {
        throw std::runtime_error("can't rot with sizes " + std::to_string(n) +
                                 " and " + std::to_string(y.size()));
    }

    const auto fn = detail::overload<T>::fn(
        cublasSrot, cublasDrot, detail::not_callable{}, detail::not_callable{});

    const auto code = fn(cublas_handle().value(),
                         n,
                         x.data(),
                         detail::incx(),
                         y.data(),
                         detail::incy(),
                         &c,
                         &s);

    check_error(code);
}

template <template <typename> class DeviceStorage, typename T>
inline void scal(const T alpha, DeviceStorage<T>& x)
{
    const auto fn = detail::overload<T>::fn(
        cublasSscal, cublasDscal, cublasCscal, cublasZscal);

    const auto code =
        fn(cublas_handle().value(), x.size(), &alpha, x.data(), detail::incx());

    check_error(code);
}

} // namespace duda

#endif /* DUDA_BLAS_LEVEL1_HPP_ */
