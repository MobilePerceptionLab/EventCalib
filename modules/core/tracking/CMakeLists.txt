add_library(tracking SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/TrackingBase.cpp
        )
add_library(opengv2::tracking ALIAS tracking)

target_include_directories(tracking PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/tracking>
        )

target_link_libraries(tracking PUBLIC
        frame
        match
        map
        )