#ifndef DUDA_MATH_LOG_HPP_
#define DUDA_MATH_LOG_HPP_

namespace duda
{

void log(float* const data, const int size);

void log(double* const data, const int size);

template <typename DeviceStorage>
inline DeviceStorage log(DeviceStorage x)
{
    log(x.data(), x.size());

    return x;
}

} // namespace duda

#endif /* DUDA_MATH_LOG_HPP_ */
