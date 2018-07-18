#ifndef DUDA_RANDOM_HPP_
#define DUDA_RANDOM_HPP_

#include <curand.h>

#include "check_error.hpp"
#include "curand_generator.hpp"
#include "detail.hpp"

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

    check_curand_error(code);
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

    check_curand_error(code);
}

} // namespace duda

#endif /* DUDA_RANDOM_HPP_ */
