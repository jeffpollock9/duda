#ifndef DUDA_UTILITY_CUDA_STREAM_HPP_
#define DUDA_UTILITY_CUDA_STREAM_HPP_

#include <duda/utility/check_error.hpp>

#include <cuda_runtime_api.h>

namespace duda
{

struct cuda_stream_wrapper
{
    cuda_stream_wrapper()
    {
        check_error(cudaStreamCreate(&stream_));
    }

    ~cuda_stream_wrapper()
    {
        check_error(cudaStreamDestroy(stream_));
    }

    cudaStream_t& value()
    {
        return stream_;
    }

private:
    cudaStream_t stream_;
};

inline cuda_stream_wrapper& cuda_stream()
{
    static cuda_stream_wrapper stream;
    return stream;
}

} // namespace duda

#endif /* DUDA_UTILITY_CUDA_STREAM_HPP_ */
