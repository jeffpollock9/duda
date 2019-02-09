#ifndef DUDA_RANDOM_LOG_NORMAL_HPP_
#define DUDA_RANDOM_LOG_NORMAL_HPP_

#include <duda/utility/check_error.hpp>
#include <duda/detail/overload.hpp>
#include <duda/device_matrix.hpp>
#include <duda/device_vector.hpp>
#include <duda/random/curand_generator.hpp>

#include <curand.h>

namespace duda
{

template <typename T>
inline void fill_random_log_normal(T* const data,
                                   const int size,
                                   const T mean   = 0.0,
                                   const T stddev = 0.0)
{
    const auto fn = detail::overload<T>::fn(curandGenerateLogNormal,
                                            curandGenerateLogNormalDouble,
                                            detail::not_callable{},
                                            detail::not_callable{});

    const auto code = fn(curand_generator().value(), data, size, mean, stddev);

    check_error(code);
}

template <typename T>
inline device_matrix<T> random_log_normal(const int rows,
                                          const int cols,
                                          const T mean   = 0.0,
                                          const T stddev = 1.0)
{
    device_matrix<T> x(rows, cols);

    fill_random_log_normal(x.data(), x.size(), mean, stddev);

    return x;
}

template <typename T>
inline device_vector<T>
random_log_normal(const int size, const T mean = 0.0, const T stddev = 1.0)
{
    device_matrix<T> x(size);

    fill_random_log_normal(x.data(), x.size(), mean, stddev);

    return x;
}

} // namespace duda

#endif /* DUDA_RANDOM_LOG_NORMAL_HPP_ */
