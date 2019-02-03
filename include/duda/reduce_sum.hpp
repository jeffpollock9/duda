#ifndef DUDA_REDUCE_SUM_HPP_
#define DUDA_REDUCE_SUM_HPP_

namespace duda
{

int reduce_sum(const int* const data, const int size);

float reduce_sum(const float* const data, const int size);

double reduce_sum(const double* const data, const int size);

template <template <typename> class DeviceStorage, typename T>
inline T reduce_sum(const DeviceStorage<T>& x)
{
    return reduce_sum(x.data(), x.size());
}

} // namespace duda

#endif /* DUDA_REDUCE_SUM_HPP_ */
