#include <duda/detail/str.hpp>
#include <duda/device_vector.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using vector = duda::device_vector<float>;

void init_device_vector(py::module& m)
{
    py::class_<vector>(m, "DeviceVector")
        .def(py::init<int>())
        .def("__str__", &duda::detail::str<vector>);
}
