#ifndef DUDA_MATH_EXP_HPP_
#define DUDA_MATH_EXP_HPP_

namespace duda
{

void exp(float* const data, const int size);

void exp(double* const data, const int size);

template <typename DeviceStorage>
inline DeviceStorage exp(DeviceStorage x)
{
    exp(x.data(), x.size());

    return x;
}

} // namespace duda

#endif /* DUDA_MATH_EXP_HPP_ */
