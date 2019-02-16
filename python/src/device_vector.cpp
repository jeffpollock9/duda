#include <duda/device_vector.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename T>
std::string str(const T& t)
{
    std::ostringstream os;
    os << t;
    return os.str();
}

void init_device_vector(py::module& m)
{
    py::class_<duda::device_vector<double>>(m, "DeviceVector")
        .def(py::init<int>())
        .def("__str__",
             [](const duda::device_vector<double>& self) { return str(self); });
}
