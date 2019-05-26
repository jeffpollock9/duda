#include <duda/memory_manager.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_memory_manager(py::module& m)
{
    py::class_<duda::memory_manager>(m, "MemoryManager")
        .def(py::init<duda::allocation_mode, std::size_t, bool>(),
             py::arg("allocation") = duda::allocation_mode::pool_allocation,
             py::arg("initial_pool_size") = 0,
             py::arg("enable_logging")    = false);
}
