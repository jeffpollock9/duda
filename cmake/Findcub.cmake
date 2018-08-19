find_path(cub_INCLUDE_DIR
    NAMES cub/cub.cuh
    DOC "cub header only library")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cub REQUIRED_VARS cub_INCLUDE_DIR)

if(cub_FOUND)
    set(cub_INCLUDE_DIRS ${cub_INCLUDE_DIR})
    if(NOT TARGET cub::cub)
        add_library(cub::cub INTERFACE IMPORTED)
        set_target_properties(cub::cub PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${cub_INCLUDE_DIR}")
    endif()
endif()

mark_as_advanced(cub_INCLUDE_DIR)
