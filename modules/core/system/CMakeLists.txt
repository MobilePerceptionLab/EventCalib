add_library(system SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SystemBase.cpp
        )
add_library(opengv2::system ALIAS system)

target_include_directories(system PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/system>
        )

target_link_libraries(system PUBLIC
        map
        viewer
        map_optimizer
        tracking
        feature_extractor
        match
        )