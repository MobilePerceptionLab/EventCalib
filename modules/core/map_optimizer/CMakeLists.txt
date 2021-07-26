add_library(map_optimizer SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/MapOptimizerBase.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/GVMapOptimizer.cpp
        )
add_library(opengv2::map_optimizer ALIAS map_optimizer)

target_include_directories(map_optimizer PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/map_optimizer>
        )

target_link_libraries(map_optimizer PUBLIC
        map
        viewer
        bundle_adjustment
        )