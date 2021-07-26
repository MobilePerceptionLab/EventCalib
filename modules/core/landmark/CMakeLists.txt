add_library(landmark INTERFACE)
add_library(opengv2::landmark ALIAS landmark)

target_include_directories(landmark INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/landmark>
        )

target_link_libraries(landmark INTERFACE
        feature
        )