#ifndef DEVICE_VECTOR_HPP_
#define DEVICE_VECTOR_HPP_

#include <duda/detail/value_proxy.hpp>
#include <duda/utility/check_error.hpp>
#include <duda/utility/copy.hpp>
#include <duda/utility/print_precision.hpp>

#include <cuda_runtime_api.h>

#include <iomanip>
#include <iosfwd>
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
        copy_host_to_device(host, *this);
    }

    device_vector(const device_vector& x) : device_vector(x.size())
    {
        copy_device_to_device(x, *this);
    }

    device_vector(device_vector&& x) noexcept : size_(x.size())
    {
        std::swap(data_, x.data_);
    }

    device_vector& operator=(device_vector x)
    {
        size_ = x.size();

        std::swap(data_, x.data_);

        return *this;
    }

    ~device_vector()
    {
        check_error(cudaFree(data_));
    }

    const T operator()(const int index) const
    {
        T value;

        const auto code = cudaMemcpy(
            &value, data_ + index, sizeof(T), cudaMemcpyDeviceToHost);

        check_error(code);

        return value;
    }

    detail::value_proxy<T> operator()(const int index)
    {
        return {data_, index};
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

template <typename T>
std::ostream& operator<<(std::ostream& os, const device_vector<T>& x)
{
    std::ios flags(nullptr);
    flags.copyfmt(os);

    const auto precision = print_precision().value();

    os << std::setprecision(precision) << std::scientific << std::showpos
       << "[ ";

    for (int i = 0; i < x.size(); ++i)
    {
        os << x(i) << " ";
    }

    os << "]\n";

    os.copyfmt(flags);

    return os;
}

} // namespace duda

#endif /* DEVICE_VECTOR_HPP_ */
