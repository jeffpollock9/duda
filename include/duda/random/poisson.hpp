#ifndef DUDA_RANDOM_POISSON_HPP_
#define DUDA_RANDOM_POISSON_HPP_

#include <duda/utility/check_error.hpp>
#include <duda/detail/overload.hpp>
#include <duda/device_matrix.hpp>
#include <duda/device_vector.hpp>
#include <duda/random/curand_generator.hpp>

#include <curand.h>

namespace duda
{

inline void fill_random_poisson(unsigned int* const data,
                                const int size,
                                const double lambda)
{
    const auto code =
        curandGeneratePoisson(curand_generator().value(), data, size, lambda);

    check_error(code);
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

#endif /* DUDA_RANDOM_POISSON_HPP_ */
