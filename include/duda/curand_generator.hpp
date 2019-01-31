#ifndef DUDA_CURAND_GENERATOR_HPP_
#define DUDA_CURAND_GENERATOR_HPP_

#include <duda/check_error.hpp>

#include <curand.h>

namespace duda
{

struct curand_generator_wrapper
{
    curand_generator_wrapper()
    {
        check_error(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    }

    ~curand_generator_wrapper()
    {
        check_error(curandDestroyGenerator(gen_));
    }

    curandGenerator_t& value()
    {
        return gen_;
    }

    void seed(const unsigned long long seed   = 0xdeadbeef,
              const unsigned long long offset = 0)
    {
        check_error(curandSetPseudoRandomGeneratorSeed(gen_, seed));
        check_error(curandSetGeneratorOffset(gen_, offset));
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
