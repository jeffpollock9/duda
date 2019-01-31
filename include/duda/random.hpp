#ifndef DUDA_RANDOM_HPP_
#define DUDA_RANDOM_HPP_

#include <duda/check_error.hpp>
#include <duda/curand_generator.hpp>
#include <duda/detail.hpp>
#include <duda/device_matrix.hpp>
#include <duda/device_vector.hpp>

#include <curand.h>

namespace duda
{

template <typename T>
inline void fill_random_uniform(T* const data, const int size)
{
    const auto code = detail::overload<T>::call(curandGenerateUniform,
                                                curandGenerateUniformDouble,
                                                curand_gen().value(),
                                                data,
                                                size);

    check_error(code);
}

template <typename T>
inline void fill_random_normal(T* const data,
                               const int size,
                               const T mean   = 0.0,
                               const T stddev = 0.0)
{
    const auto code = detail::overload<T>::call(curandGenerateNormal,
                                                curandGenerateNormalDouble,
                                                curand_gen().value(),
                                                data,
                                                size,
                                                mean,
                                                stddev);

    check_error(code);
}

template <typename T>
inline void fill_random_log_normal(T* const data,
                                   const int size,
                                   const T mean   = 0.0,
                                   const T stddev = 0.0)
{
    const auto code = detail::overload<T>::call(curandGenerateLogNormal,
                                                curandGenerateLogNormalDouble,
                                                curand_gen().value(),
                                                data,
                                                size,
                                                mean,
                                                stddev);

    check_error(code);
}

inline void fill_random_poisson(unsigned int* const data,
                                const int size,
                                const double lambda)
{
    const auto code =
        curandGeneratePoisson(curand_gen().value(), data, size, lambda);

    check_error(code);
}

template <typename T>
inline device_matrix<T> random_uniform(const int rows, const int cols)
{
    device_matrix<T> x(rows, cols);

    fill_random_uniform(x.data(), x.size());

    return x;
}

template <typename T>
inline device_vector<T> random_uniform(const int size)
{
    device_vector<T> x(size);

    fill_random_uniform(x.data(), x.size());

    return x;
}

template <typename T>
inline device_matrix<T> random_normal(const int rows,
                                      const int cols,
                                      const T mean   = 0.0,
                                      const T stddev = 1.0)
{
    device_matrix<T> x(rows, cols);

    fill_random_normal(x.data(), x.size(), mean, stddev);

    return x;
}

template <typename T>
inline device_vector<T>
random_normal(const int size, const T mean = 0.0, const T stddev = 1.0)
{
    device_vector<T> x(size);

    fill_random_normal(x.data(), x.size(), mean, stddev);

    return x;
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

inline device_matrix<unsigned int>
random_poisson(const int rows, const int cols, const double lambda)
{
    device_matrix<unsigned int> x(rows, cols);

    fill_random_poisson(x.data(), x.size(), lambda);

    return x;
}

inline device_vector<unsigned int> random_poisson(const int size,
                                                  const double lambda)
{
    device_vector<unsigned int> x(size);

    fill_random_poisson(x.data(), x.size(), lambda);

    return x;
}

} // namespace duda

#endif /* DUDA_RANDOM_HPP_ */
