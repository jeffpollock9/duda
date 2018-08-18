#include "device_matrix.hpp"

namespace duda
{

template <typename T>
inline void copy(const device_matrix<T>& device, T* const host)
{
    const auto code =
        cudaMemcpy(host, device.data(), device.bytes(), cudaMemcpyDeviceToHost);

    check_error(code);
}

} // namespace duda
