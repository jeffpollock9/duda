#ifndef DUDA_BENCHMARK_HELPERS_HPP_
#define DUDA_BENCHMARK_HELPERS_HPP_

#include "device_matrix.hpp"

#include "Eigen/Dense"
#include "benchmark/benchmark.h"

#define DUDA_BENCHMARK_RANGE Range(8, 8 << 8)

template <typename T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using device_matrix = duda::device_matrix<T>;

#endif /* DUDA_BENCHMARK_HELPERS_HPP_ */
