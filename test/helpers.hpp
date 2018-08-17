#ifndef DUDA_TEST_HELPERS_HPP_
#define DUDA_TEST_HELPERS_HPP_

#include "eye.hpp"
#include "fill.hpp"
#include "blas.hpp"
#include "copy.hpp"
#include "random.hpp"
#include "reduce_sum.hpp"
#include "device_matrix.hpp"

#include "Eigen/Dense"
#include "catch/catch.hpp"

template <typename T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using device_matrix = duda::device_matrix<T>;

template <typename T>
host_matrix<T> copy(const device_matrix<T>& device)
{
    host_matrix<T> host(device.rows(), device.cols());

    duda::copy(device, host.data());

    return host;
}

template <typename T>
device_matrix<T> copy(const host_matrix<T>& host)
{
    const int rows = host.rows();
    const int cols = host.cols();

    return {host.data(), rows, cols};
}

#endif /* DUDA_TEST_HELPERS_HPP_ */
