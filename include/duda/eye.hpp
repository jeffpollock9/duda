#ifndef DUDA_EYE_HPP_
#define DUDA_EYE_HPP_

#include <duda/device_matrix.hpp>

namespace duda
{

void eye(int* const data, const int dim);

void eye(float* const data, const int dim);

void eye(double* const data, const int dim);

template <typename T>
inline device_matrix<T> eye(const int dim)
{
    device_matrix<T> x(dim, dim);

    eye(x.data(), dim);

    return x;
}

} // namespace duda

#endif /* DUDA_EYE_HPP_ */
