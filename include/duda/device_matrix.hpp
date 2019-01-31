#ifndef DEVICE_MATRIX_HPP_
#define DEVICE_MATRIX_HPP_

#include <duda/check_error.hpp>

#include <cuda_runtime_api.h>

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
        check_error(cudaMemcpy(data_, host, bytes(), cudaMemcpyHostToDevice));
    }

    device_matrix(const device_matrix& x) : device_matrix(x.rows(), x.cols())
    {
        check_error(
            cudaMemcpy(data_, x.data_, bytes(), cudaMemcpyDeviceToDevice));
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

} // namespace duda

#endif /* DEVICE_MATRIX_HPP_ */
