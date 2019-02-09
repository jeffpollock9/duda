#ifndef TESTING_HPP_
#define TESTING_HPP_

#include <duda/device_matrix.hpp>
#include <duda/device_vector.hpp>
#include <duda/utility/copy.hpp>

#include <Eigen/Dense>
#include <catch.hpp>

namespace testing
{

template <typename T>
using host_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
inline host_vector<T> copy(const duda::device_vector<T>& device)
{
    host_vector<T> host(device.size());

    duda::copy_device_to_host(device, host.data());

    return host;
}

template <typename T>
inline host_matrix<T> copy(const duda::device_matrix<T>& device)
{
    host_matrix<T> host(device.rows(), device.cols());

    duda::copy_device_to_host(device, host.data());

    return host;
}

template <typename T>
inline duda::device_matrix<T> copy(const host_matrix<T>& host)
{
    const int rows = host.rows();
    const int cols = host.cols();

    return {host.data(), rows, cols};
}

template <typename T>
inline duda::device_vector<T> copy(const host_vector<T>& host)
{
    const int size = host.size();

    return {host.data(), size};
}

template <typename DeviceStorage, typename HostStorage>
inline bool all_close(const DeviceStorage& device, const HostStorage& host)
{
    const auto device_data_on_host = copy(device);

    return host.isApprox(device_data_on_host);
}

} // namespace testing

#endif /* TESTING_HPP_ */
