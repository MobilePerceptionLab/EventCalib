add_library(simulator SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Simulator.cpp
        )
add_library(opengv2::simulator ALIAS simulator)

target_include_directories(simulator PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/simulator>
        )

target_link_libraries(simulator PUBLIC
        map
        bundle_adjustment
        )