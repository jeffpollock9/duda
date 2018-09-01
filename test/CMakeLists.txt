include(add_basic_warnings)

find_package(Eigen3 REQUIRED)

add_library(catch SHARED catch/catch.cpp)

set(DUDA_TESTS
    device_matrix
    device_vector
    blas/level1
    blas/level2
    blas/level3
    eye
    fill
    reduce_sum)

foreach(test ${DUDA_TESTS})
    string(REGEX REPLACE "/" "_" name ${test})
    add_executable(test_${name} ${test}.cpp)
    target_link_libraries(test_${name} duda catch Eigen3::Eigen)
    add_basic_warnings(test_${name})
    add_test(${name} test_${name})
endforeach()
