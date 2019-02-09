#ifndef DUDA_UTILITY_COPY_DEVICE_TO_DEVICE_HPP_
#define DUDA_UTILITY_COPY_DEVICE_TO_DEVICE_HPP_

#include <duda/utility/check_error.hpp>

#include <cuda_runtime_api.h>

namespace duda
{

template <template <typename> class DeviceStorage, typename T>
inline void copy_host_to_device(const T* const host, DeviceStorage<T>& device)
{
    const auto code =
        cudaMemcpy(device.data(), host, device.bytes(), cudaMemcpyHostToDevice);

    check_error(code);
}

template <template <typename> class DeviceStorage, typename T>
inline void copy_device_to_host(const DeviceStorage<T>& device, T* const host)
{
    const auto code =
        cudaMemcpy(host, device.data(), device.bytes(), cudaMemcpyDeviceToHost);

    check_error(code);
}

template <typename DeviceStorage>
inline void copy_device_to_device(const DeviceStorage& in, DeviceStorage& out)
{
    const auto code =
        cudaMemcpy(out.data(), in.data(), in.bytes(), cudaMemcpyDeviceToDevice);

    check_error(code);
}

} // namespace duda

#endif /* DUDA_UTILITY_COPY_DEVICE_TO_DEVICE_HPP_ */
