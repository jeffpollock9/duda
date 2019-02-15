#ifndef DUDA_DETAIL_VALUE_PROXY_HPP_
#define DUDA_DETAIL_VALUE_PROXY_HPP_

#include <duda/utility/check_error.hpp>

#include <cuda_runtime_api.h>

namespace duda
{

namespace detail
{

template <typename T>
struct value_proxy
{
    void operator=(const T value)
    {
        const auto code =
            cudaMemcpy(data + index, &value, sizeof(T), cudaMemcpyHostToDevice);

        check_error(code);
    }

    T* data;
    int index;
};

} // namespace detail

} // namespace duda

#endif /* DUDA_DETAIL_VALUE_PROXY_HPP_ */
