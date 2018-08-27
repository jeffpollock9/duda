#ifndef DEVICE_VECTOR_HPP_
#define DEVICE_VECTOR_HPP_

#include "check_error.hpp"

#include <cuda_runtime_api.h>

#include <utility>

namespace duda
{

template <typename T>
struct device_vector
{
    device_vector() = default;

    device_vector(const int size) : size_(size)
    {
        check_error(cudaMalloc((void**)&data_, bytes()));
    }

    device_vector(const T* const host, const int size) : device_vector(size)
    {
        check_error(cudaMemcpy(data_, host, bytes(), cudaMemcpyHostToDevice));
    }

    device_vector(const device_vector& x) : device_vector(x.size())
    {
        check_error(
            cudaMemcpy(data_, x.data_, bytes(), cudaMemcpyDeviceToDevice));
    }

    device_vector(device_vector&& x) : size_(x.size())
    {
        std::swap(data_, x.data_);
    }

    device_vector& operator=(device_vector x)
    {
        size_ = x.size();

        std::swap(data_, x.data_);

        return *this;
    }

    device_vector& operator=(device_vector&& x)
    {
        size_ = x.size();

        std::swap(data_, x.data_);

        return *this;
    }

    ~device_vector()
    {
        check_error(cudaFree(data_));
    }

    int size() const
    {
        return size_;
    }

    int bytes() const
    {
        return size() * sizeof(T);
    }

    T* data()
    {
        return data_;
    }

    const T* data() const
    {
        return data_;
    }

private:
    T* data_  = nullptr;
    int size_ = 0;
};

} // namespace duda

#endif /* DEVICE_VECTOR_HPP_ */
