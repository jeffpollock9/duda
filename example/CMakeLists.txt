set(DUDA_EXAMPLES print)

foreach(example ${DUDA_EXAMPLES})
    add_executable(example_${example} ${example}.cpp)
    target_link_libraries(example_${example} duda)
endforeach()
