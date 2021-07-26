# ----------------------------------------------------------------------------
# Root CMake file for opengv2
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.1)
project(opengv2 LANGUAGES CXX VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELEASE "-O3")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

LIST(APPEND CMAKE_MODULE_PATH cmake/cmake_modules)

include(GNUInstallDirs)
# Set relative install directories
set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
set(INSTALL_PKGCONFIG_DIR "${CMAKE_INSTALL_DATADIR}/pkgconfig")
set(INSTALL_CMAKE_DIR "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}")
set(INSTALL_COPYRIGHT_DIR "${CMAKE_INSTALL_DOCDIR}")

# configure a header file to pass some of the CMake settings to the source code
configure_file(cmake/CmakeConfig.h.in CmakeConfig.h)
# include 'CmakeConfig.h'
include_directories(${PROJECT_BINARY_DIR})

# http://eigen.tuxfamily.org/dox/TopicPitfalls.html
# recommend Ref<T> for a writable reference, and const Ref<const T>& for a const reference.
# there is no problem with functions that return objects by value.
# use Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> with std containers
# Don't use auto with Eigen type.
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(nanoflann REQUIRED)
# TODO: add serialization functionality by cereal
#find_package(matplotlib_cpp) # plot library, matplotlib_cpp::matplotlib_cpp

add_subdirectory(modules/core)
add_subdirectory(modules/camera_calibration/event)
add_subdirectory(modules/camera_calibration/event_camera_calib)
add_subdirectory(modules/camera_calibration/dbscan)
add_subdirectory(modules/camera_calibration/cv_calib)
