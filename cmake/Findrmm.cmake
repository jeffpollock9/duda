find_path(rmm_INCLUDE_DIR
    NAMES rmm/rmm.h
    DOC "rmm include directory")

find_library(rmm_LIBRARY
    NAMES lib/librmm.so
    DOC "rmm shared library")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rmm REQUIRED_VARS rmm_INCLUDE_DIR rmm_LIBRARY)

if(rmm_FOUND)
    if(NOT TARGET rmm::rmm)
        add_library(rmm::rmm INTERFACE IMPORTED)
        set_target_properties(rmm::rmm PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${rmm_INCLUDE_DIR}
            INTERFACE_LINK_LIBRARIES ${rmm_LIBRARY})
    endif()
endif()
