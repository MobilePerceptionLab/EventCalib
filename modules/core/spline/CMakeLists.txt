# Each rotation is represented by two unit quaternions of opposite sign, which will affect the continuity of values,
# remember that for rotation spline.

add_library(spline SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/BsplineSO3.cpp
        )
add_library(opengv2::spline ALIAS spline)

target_include_directories(spline PUBLIC
        ${Sophus_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/spline>
        )

target_link_libraries(spline PUBLIC
        ${CERES_LIBRARIES}
        )

add_executable(BsplineSO3_test ${CMAKE_CURRENT_SOURCE_DIR}/test/BsplineSO3_test.cpp)
target_link_libraries(BsplineSO3_test PUBLIC
        spline
        system
        simulator
        bundle_adjustment
        )