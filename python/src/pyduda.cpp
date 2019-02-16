#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_device_matrix(py::module&);
void init_device_vector(py::module&);
void init_random(py::module&);

PYBIND11_MODULE(example, m)
{
    init_device_matrix(m);
    init_device_vector(m);
    init_random(m);
}
