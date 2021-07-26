# EventCameraCalib, TODO: multi-spline over different direction (such as PTAM)
add_library(event_camera_calib SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/CirclesEventFrame.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/EventCalibIni.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/EventCalibSpline.cpp
        )
add_library(opengv2::event_camera_calib ALIAS event_camera_calib)

target_include_directories(event_camera_calib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/event_camera_calib>
        )

target_link_libraries(event_camera_calib PUBLIC
        core
        cv_calib
        event
        dbscan
        )

# test
add_executable(unit_test_eventCameraCalib ${CMAKE_CURRENT_SOURCE_DIR}/test/eventCameraCalib.cpp)
target_link_libraries(unit_test_eventCameraCalib PUBLIC
        event_camera_calib
        core
        stdc++fs
        )

add_executable(unit_test_inverseDistortion ${CMAKE_CURRENT_SOURCE_DIR}/test/unit_test_inverseDistortion.cpp)
target_link_libraries(unit_test_inverseDistortion PUBLIC
        core)

add_executable(handEyeCalib ${CMAKE_CURRENT_SOURCE_DIR}/test/HandEyeCalibration.cpp)
target_link_libraries(handEyeCalib PUBLIC
        core)

add_executable(visulization_eventCalib ${CMAKE_CURRENT_SOURCE_DIR}/test/visulization_eventCalib.cpp)
target_link_libraries(visulization_eventCalib PUBLIC
        core
        stdc++fs)