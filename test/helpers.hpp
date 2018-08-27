#ifndef DUDA_TEST_HELPERS_HPP_
#define DUDA_TEST_HELPERS_HPP_

#include "copy.hpp"
#include "device_matrix.hpp"
#include "device_vector.hpp"

#include "Eigen/Dense"
#include "catch/catch.hpp"

template <typename T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using device_matrix = duda::device_matrix<T>;

template <typename T>
using host_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using device_vector = duda::device_vector<T>;

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

template <typename T>
host_vector<T> copy(const device_vector<T>& device)
{
    host_vector<T> host(device.size());

    duda::copy(device, host.data());

    return host;
}

template <typename T>
device_vector<T> copy(const host_vector<T>& host)
{
    const int size = host.size();

    return {host.data(), size};
}

#endif /* DUDA_TEST_HELPERS_HPP_ */
