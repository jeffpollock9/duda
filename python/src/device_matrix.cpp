#include <duda/detail/str.hpp>
#include <duda/device_matrix.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

using matrix = duda::device_matrix<float>;

void init_device_matrix(py::module& m)
{
    py::class_<matrix>(m, "DeviceMatrix")
        .def(py::init<int, int>())
        .def("__str__", &duda::detail::str<matrix>);
}
