#include <duda/utility/enums.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

#define DUDA_ENUM_VALUE(e, v) value(#v, e::v)

void init_enums(py::module& m)
{
    using duda::allocation_mode;
    using duda::fill_mode;
    using duda::op;

    py::enum_<allocation_mode>(m, "AllocationMode")
        .DUDA_ENUM_VALUE(allocation_mode, cuda_default_allocation)
        .DUDA_ENUM_VALUE(allocation_mode, pool_allocation)
        .DUDA_ENUM_VALUE(allocation_mode, cuda_managed_memory);

    py::enum_<fill_mode>(m, "FillMode")
        .DUDA_ENUM_VALUE(fill_mode, lower)
        .DUDA_ENUM_VALUE(fill_mode, upper);

    py::enum_<op>(m, "Op")
        .DUDA_ENUM_VALUE(op, none)
        .DUDA_ENUM_VALUE(op, transpose)
        .DUDA_ENUM_VALUE(op, conjugate_transpose);
}
