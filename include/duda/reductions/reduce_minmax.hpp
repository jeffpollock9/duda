#ifndef DUDA_REDUCTIONS_REDUCE_MINMAX_HPP_
#define DUDA_REDUCTIONS_REDUCE_MINMAX_HPP_

#include <utility>

namespace duda
{

std::pair<int, int> reduce_minmax(const int* const data, const int size);

std::pair<float, float> reduce_minmax(const float* const data, const int size);

std::pair<double, double> reduce_minmax(const double* const data, const int size);

template <template <typename> class DeviceStorage, typename T>
inline std::pair<T, T> reduce_minmax(const DeviceStorage<T>& x)
{
    return reduce_minmax(x.data(), x.size());
}

} // namespace duda

#endif /* DUDA_REDUCTIONS_REDUCE_MINMAX_HPP_ */
