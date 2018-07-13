#ifndef DUDA_RANDOM_HPP_
#define DUDA_RANDOM_HPP_

#include <curand.h>

#include "check_error.hpp"
#include "curand_generator.hpp"

namespace duda
{

inline void fill_random_uniform(float* const data, const int size)
{
    const auto code = curandGenerateUniform(curand_gen().value(), data, size);

    check_curand_error(code);
}

inline void fill_random_uniform(double* const data, const int size)
{
    const auto code =
        curandGenerateUniformDouble(curand_gen().value(), data, size);

    check_curand_error(code);
}

} // namespace duda

#endif /* DUDA_RANDOM_HPP_ */
