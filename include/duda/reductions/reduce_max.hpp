#ifndef DUDA_REDUCTIONS_REDUCE_MAX_HPP_
#define DUDA_REDUCTIONS_REDUCE_MAX_HPP_

namespace duda
{

int reduce_max(const int* const data, const int size);

float reduce_max(const float* const data, const int size);

double reduce_max(const double* const data, const int size);

template <template <typename> class DeviceStorage, typename T>
inline T reduce_max(const DeviceStorage<T>& x)
{
    return reduce_max(x.data(), x.size());
}

} // namespace duda

#endif /* DUDA_REDUCTIONS_REDUCE_MAX_HPP_ */
