#include <duda/linalg/matmul.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_matmul(py::module& m)
{
    m.def("matmul",
          &duda::matmul<float>,
          py::arg("a"),
          py::arg("b"),
          py::arg("op_a")  = duda::op::none,
          py::arg("op_b")  = duda::op::none,
          py::arg("alpha") = 1.0f);
}
