add_library(map SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/MapBase.cpp
        )
add_library(opengv2::map ALIAS map)

target_include_directories(map PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/map>
        )

target_link_libraries(map PUBLIC
        landmark
        frame
        )