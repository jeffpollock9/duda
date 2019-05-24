find_path(cub_INCLUDE_DIR
    NAMES cub/cub.cuh
    DOC "cub include directory")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cub REQUIRED_VARS cub_INCLUDE_DIR)

if(cub_FOUND)
    if(NOT TARGET cub::cub)
        add_library(cub::cub INTERFACE IMPORTED)
        set_target_properties(cub::cub PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${cub_INCLUDE_DIR})
    endif()
endif()
