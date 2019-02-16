#include <duda/random.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_random(py::module& m)
{
    m.def("random_uniform",
          py::overload_cast<int, int>(&duda::random_uniform<float>),
          "random uniform for a matrix");

    m.def("random_uniform",
          py::overload_cast<int>(&duda::random_uniform<float>),
          "random uniform for a vector");
}
