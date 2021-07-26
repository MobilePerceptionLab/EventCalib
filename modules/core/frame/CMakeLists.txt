add_library(frame INTERFACE)
add_library(opengv2::frame ALIAS frame)

target_include_directories(frame INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/frame>
        )

target_link_libraries(frame INTERFACE
        sensor
        feature
        feature_extractor
        utility
        )