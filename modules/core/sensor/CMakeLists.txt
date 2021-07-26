add_library(sensor SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/OmniScaramuzzaCamera.cpp
        #${CMAKE_CURRENT_SOURCE_DIR}/src/OmniMeiCamera.cpp, not tested
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PinholeCamera.cpp
        )
add_library(opengv2::sensor ALIAS sensor)

target_include_directories(sensor PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/sensor>
        )

target_link_libraries(sensor PUBLIC
        utility
        ${OpenCV_LIBS}
        )