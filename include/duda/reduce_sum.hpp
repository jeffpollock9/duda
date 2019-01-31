#ifndef DUDA_REDUCE_SUM_HPP_
#define DUDA_REDUCE_SUM_HPP_

#include <duda/device_matrix.hpp>

namespace duda
{

int reduce_sum(const int* const data, const int size);

float reduce_sum(const float* const data, const int size);

double reduce_sum(const double* const data, const int size);

template <template <typename> class Device, typename T>
inline T reduce_sum(const Device<T>& x)
{
    return reduce_sum(x.data(), x.size());
}

} // namespace duda

#endif /* DUDA_REDUCE_SUM_HPP_ */
