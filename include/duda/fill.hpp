#ifndef DUDA_FILL_HPP_
#define DUDA_FILL_HPP_

namespace duda
{

void fill(int* const data, const int size, const int value);

void fill(float* const data, const int size, const float value);

void fill(double* const data, const int size, const double value);

template <template <typename> class DeviceStorage, typename T>
inline void fill(DeviceStorage<T>& x, const T value)
{
    fill(x.data(), x.size(), value);
}

} // namespace duda

#endif /* DUDA_FILL_HPP_ */
