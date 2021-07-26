add_library(utility INTERFACE)
add_library(opengv2::utility ALIAS utility)

target_include_directories(utility INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/utility>
        )

target_link_libraries(utility INTERFACE
        nanoflann::nanoflann
        )