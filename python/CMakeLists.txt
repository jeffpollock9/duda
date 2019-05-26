include(add_basic_warnings)

find_package(pybind11 REQUIRED)

set(DUDA_PYTHON
    src/pyduda.cpp
    src/device_matrix.cpp
    src/device_vector.cpp
    src/random.cpp
    src/enums.cpp
    src/memory_manager.cpp
    src/matmul.cpp
    )

pybind11_add_module(pyduda ${DUDA_PYTHON})

target_link_libraries(pyduda PRIVATE duda)
