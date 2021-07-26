add_library(feature INTERFACE)
add_library(opengv2::feature ALIAS feature)

target_include_directories(feature INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/feature>
        )

target_link_libraries(feature INTERFACE
        landmark
        )