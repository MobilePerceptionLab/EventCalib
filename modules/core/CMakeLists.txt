add_subdirectory(bundle_adjustment)
add_subdirectory(feature)
add_subdirectory(feature_extractor)
add_subdirectory(frame)
add_subdirectory(landmark)
add_subdirectory(map)
add_subdirectory(map_optimizer)
add_subdirectory(match)
add_subdirectory(sensor)
add_subdirectory(spline)
add_subdirectory(system)
add_subdirectory(tracking)
add_subdirectory(simulator)
add_subdirectory(utility)
add_subdirectory(viewer)

add_library(core INTERFACE)
add_library(opengv2::core ALIAS core)
target_link_libraries(core INTERFACE
        bundle_adjustment
        feature
        feature_extractor
        frame
        landmark
        map
        map_optimizer
        match
        sensor
        spline
        system
        tracking
        simulator
        utility
        viewer
        )