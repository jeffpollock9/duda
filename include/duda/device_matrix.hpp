#ifndef DEVICE_MATRIX_HPP_
#define DEVICE_MATRIX_HPP_

#include <duda/utility/check_error.hpp>
#include <duda/utility/print_precision.hpp>
#include <duda/utility/copy.hpp>

#include <cuda_runtime_api.h>

#include <iomanip>
#include <iosfwd>
#include <utility>

namespace duda
{

template <typename T>
struct device_matrix
{
    device_matrix() = default;

    device_matrix(const int rows, const int cols) : rows_(rows), cols_(cols)
    {
        check_error(cudaMalloc((void**)&data_, bytes()));
    }

    device_matrix(const T* const host, const int rows, const int cols)
        : device_matrix(rows, cols)
    {
        copy_host_to_device(host, *this);
    }

    device_matrix(const device_matrix& x) : device_matrix(x.rows(), x.cols())
    {
        copy_device_to_device(x, *this);
    }

    device_matrix(device_matrix&& x) : rows_(x.rows()), cols_(x.cols())
    {
        std::swap(data_, x.data_);
    }

    device_matrix& operator=(device_matrix x)
    {
        rows_ = x.rows();
        cols_ = x.cols();

        std::swap(data_, x.data_);

        return *this;
    }

    device_matrix& operator=(device_matrix&& x)
    {
        rows_ = x.rows();
        cols_ = x.cols();

        std::swap(data_, x.data_);

        return *this;
    }

    ~device_matrix()
    {
        check_error(cudaFree(data_));
    }

    const T operator()(const int row, const int col) const
    {
        const int index = row + rows() * col;

        T value;

        const auto code = cudaMemcpy(
            &value, data_ + index, sizeof(T), cudaMemcpyDeviceToHost);

        check_error(code);

        return value;
    }

    void set(const int row, const int col, const T value)
    {
        const int index = row + rows() * col;

        const auto code = cudaMemcpy(
            data_ + index, &value, sizeof(T), cudaMemcpyHostToDevice);

        check_error(code);
    }

    int rows() const
    {
        return rows_;
    }

    int cols() const
    {
        return cols_;
    }

    int size() const
    {
        return rows_ * cols_;
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
    int rows_ = 0;
    int cols_ = 0;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const device_matrix<T>& x)
{
    std::ios flags(nullptr);
    flags.copyfmt(os);

    os << std::setprecision(print_precision().value()) << std::scientific;

    for (int i = 0; i < x.rows(); ++i)
    {
        os << "[ ";
        for (int j = 0; j < x.cols(); ++j)
        {
            os << x(i, j) << " ";
        }
        os << "]\n";
    }

    os.copyfmt(flags);

    return os;
}

} // namespace duda

#endif /* DEVICE_MATRIX_HPP_ */
