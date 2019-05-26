#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_device_matrix(py::module&);
void init_device_vector(py::module&);
void init_random(py::module&);
void init_enums(py::module&);
void init_memory_manager(py::module&);
void init_matmul(py::module&);

PYBIND11_MODULE(pyduda, m)
{
    init_device_matrix(m);
    init_device_vector(m);
    init_random(m);
    init_enums(m);
    init_memory_manager(m);
    init_matmul(m);
}
