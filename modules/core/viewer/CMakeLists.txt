find_package(Pangolin REQUIRED)

add_library(viewer SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ViewerBase.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PointLandmarkViewer.cpp
        )
add_library(opengv2::viewer ALIAS viewer)

target_include_directories(viewer PUBLIC
        ${Pangolin_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/viewer>
        )

target_link_libraries(viewer PUBLIC
        map
        tracking
        ${Pangolin_LIBRARIES}
        )