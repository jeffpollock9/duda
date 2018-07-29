find_package(Eigen3)

add_library(catch SHARED catch/catch.cpp)

set(DUDA_TESTS device_matrix blas eye)

foreach(test ${DUDA_TESTS})
    add_executable(${test} ${test}.cpp)
    target_link_libraries(${test} duda catch Eigen3::Eigen)
    # TODO: these flags are not portable
    target_compile_options(${test} PUBLIC -Wall -Wextra -Wpedantic -Wshadow)
    add_test(${test} ${test})
endforeach()
