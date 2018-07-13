#ifndef DUDA_CURAND_GENERATOR_HPP_
#define DUDA_CURAND_GENERATOR_HPP_

#include <curand.h>

#include "check_error.hpp"

namespace duda
{

struct curand_generator_wrapper
{
    curand_generator_wrapper()
    {
        const auto code =
            curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);

        check_curand_error(code);
    }

    ~curand_generator_wrapper()
    {
        const auto code = curandDestroyGenerator(gen_);

        check_curand_error(code);
    }

    curandGenerator_t& value()
    {
        return gen_;
    }

    void seed(const unsigned long long seed   = 0xdeadbeef,
              const unsigned long long offset = 0)
    {
        const auto seed_code = curandSetPseudoRandomGeneratorSeed(gen_, seed);

        check_curand_error(seed_code);

        const auto offset_code = curandSetGeneratorOffset(gen_, offset);

        check_curand_error(offset_code);
    }

private:
    curandGenerator_t gen_;
};

inline curand_generator_wrapper& curand_gen()
{
    static curand_generator_wrapper gen;
    return gen;
}

} // namespace duda

#endif /* DUDA_CURAND_GENERATOR_HPP_ */
