#ifndef DUDA_BENCHMARK_HELPERS_HPP_
#define DUDA_BENCHMARK_HELPERS_HPP_

#include "device_matrix.hpp"
#include "device_vector.hpp"

#include "Eigen/Dense"
#include "benchmark/benchmark.h"

#define DUDA_BENCHMARK_RANGE Range(8, 8 << 8)

template <typename T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using device_matrix = duda::device_matrix<T>;

template <typename T>
using host_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using device_vector = duda::device_vector<T>;

#endif /* DUDA_BENCHMARK_HELPERS_HPP_ */
