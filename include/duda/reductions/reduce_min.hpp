#ifndef DUDA_REDUCTIONS_REDUCE_MIN_HPP_
#define DUDA_REDUCTIONS_REDUCE_MIN_HPP_

namespace duda
{

int reduce_min(const int* const data, const int size);

float reduce_min(const float* const data, const int size);

double reduce_min(const double* const data, const int size);

template <template <typename> class DeviceStorage, typename T>
inline T reduce_min(const DeviceStorage<T>& x)
{
    return reduce_min(x.data(), x.size());
}

} // namespace duda

#endif /* DUDA_REDUCTIONS_REDUCE_MIN_HPP_ */
