add_library(feature_extractor SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cvPointExtractor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/orbPointExtractor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ORBextractor.cc
        )
add_library(opengv2::feature_extractor ALIAS feature_extractor)

target_include_directories(feature_extractor PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/feature_extractor>
        )

target_link_libraries(feature_extractor PUBLIC
        feature
        ${OpenCV_LIBS}
        )