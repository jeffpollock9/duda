# TEMP add rmm example stuff here

set(DUDA_EXAMPLES print rmm)

foreach(example ${DUDA_EXAMPLES})
    add_executable(example_${example} ${example}.cpp)
    target_link_libraries(example_${example} duda)
endforeach()

target_link_libraries(example_rmm rmm::rmm)
