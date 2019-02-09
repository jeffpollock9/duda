#ifndef DUDA_UTILITY_COPY_DEVICE_TO_HOST_HPP_
#define DUDA_UTILITY_COPY_DEVICE_TO_HOST_HPP_

#include <duda/device_matrix.hpp>
#include <duda/device_vector.hpp>
#include <duda/utility/check_error.hpp>

#include <cuda_runtime_api.h>

namespace duda
{

template <template <typename> class DeviceStorage, typename T>
inline void copy_device_to_host(const DeviceStorage<T>& device, T* const host)
{
    const auto code =
        cudaMemcpy(host, device.data(), device.bytes(), cudaMemcpyDeviceToHost);

    check_error(code);
}

} // namespace duda

#endif /* DUDA_UTILITY_COPY_DEVICE_TO_HOST_HPP_ */
