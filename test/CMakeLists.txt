include(add_basic_warnings)

find_package(Eigen3 REQUIRED)

add_library(testing SHARED testing/src/catch.cpp)
target_include_directories(testing PUBLIC ${PROJECT_SOURCE_DIR}/test/testing/include)
target_link_libraries(testing Eigen3::Eigen)

set(DUDA_TESTS
    device_matrix
    device_vector
    blas/level1
    blas/level2
    blas/level3
    kernels/eye
    kernels/fill
    reductions/reduce_sum)

foreach(test ${DUDA_TESTS})
    string(REGEX REPLACE "/" "_" name ${test})
    add_executable(test_${name} ${test}.cpp)
    target_link_libraries(test_${name} duda testing)
    add_basic_warnings(test_${name})
    add_test(${name} test_${name})
endforeach()
