add_library(cv_calib SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cv_calib.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/circlesgrid.cpp
        )

target_include_directories(cv_calib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/cv_calib>
        )

target_link_libraries(cv_calib PUBLIC
        ${OpenCV_LIBS}
        )

add_executable(opencv_camera_calibration ${CMAKE_CURRENT_SOURCE_DIR}/app/camera_calibration.cpp)
target_link_libraries(opencv_camera_calibration PRIVATE ${OpenCV_LIBS} sensor stdc++fs)