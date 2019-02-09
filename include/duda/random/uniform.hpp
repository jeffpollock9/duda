#ifndef DUDA_RANDOM_UNIFORM_HPP_
#define DUDA_RANDOM_UNIFORM_HPP_

#include <duda/utility/check_error.hpp>
#include <duda/detail/overload.hpp>
#include <duda/device_matrix.hpp>
#include <duda/device_vector.hpp>
#include <duda/random/curand_generator.hpp>

#include <curand.h>

namespace duda
{

template <typename T>
inline void fill_random_uniform(T* const data, const int size)
{
    const auto fn = detail::overload<T>::fn(curandGenerateUniform,
                                            curandGenerateUniformDouble,
                                            detail::not_callable{},
                                            detail::not_callable{});

    const auto code = fn(curand_generator().value(), data, size);

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

} // namespace duda

#endif /* DUDA_RANDOM_UNIFORM_HPP_ */
