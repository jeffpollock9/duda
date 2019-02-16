#include <duda/device_matrix.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename T>
std::string str(const T& t)
{
    std::ostringstream os;
    os << t;
    return os.str();
}

void init_device_matrix(py::module& m)
{
    py::class_<duda::device_matrix<double>>(m, "DeviceMatrix")
        .def(py::init<int, int>())
        .def("__str__",
             [](const duda::device_matrix<double>& self) { return str(self); });
}
