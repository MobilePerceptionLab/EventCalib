add_library(match SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/MatcherBase.cpp
        )
add_library(opengv2::match ALIAS match)

target_include_directories(match PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/match>
        )

target_link_libraries(match PUBLIC
        landmark
        frame
        ${OpenCV_LIBS}
        )