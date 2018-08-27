#include "device_matrix.hpp"

namespace duda
{

template <template <typename> class Device, typename T>
inline void copy(const Device<T>& device, T* const host)
{
    const auto code =
        cudaMemcpy(host, device.data(), device.bytes(), cudaMemcpyDeviceToHost);

    check_error(code);
}

} // namespace duda
