find_package(Eigen3 REQUIRED)

add_library(catch SHARED catch/catch.cpp)

set(DUDA_TESTS device_matrix blas eye fill reduce_sum)

foreach(test ${DUDA_TESTS})
    add_executable(test_${test} ${test}.cpp)
    target_link_libraries(test_${test} duda catch Eigen3::Eigen)
    # TODO: these flags are not portable
    target_compile_options(test_${test} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    add_test(${test} test_${test})
endforeach()
