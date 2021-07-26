add_library(bundle_adjustment SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NormalBundleAdjustment.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SplineBundleAdjustment.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SplineBundleAdjustmentV1.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SplineBundleAdjustmentV2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/RelativeConstraintBA.cpp
        )
add_library(opengv2::bundle_adjustment ALIAS bundle_adjustment)

target_include_directories(bundle_adjustment PUBLIC
        ${CERES_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/bundle_adjustment>
        )

target_link_libraries(bundle_adjustment PUBLIC
        map
        spline
        ${CERES_LIBRARIES}
        )