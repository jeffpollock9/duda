#ifndef DUDA_COPY_HPP_
#define DUDA_COPY_HPP_

#include <duda/check_error.hpp>

#include <cublas_v2.h>

namespace duda
{

template <template <typename> class DeviceStorage, typename T>
inline void copy(const DeviceStorage<T>& device, T* const host)
{
    const auto code =
        cudaMemcpy(host, device.data(), device.bytes(), cudaMemcpyDeviceToHost);

    check_error(code);
}

} // namespace duda

#endif /* DUDA_COPY_HPP_ */
