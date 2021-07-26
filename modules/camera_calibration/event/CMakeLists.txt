add_library(event SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/EventStream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/EventFrame.cpp
        )
add_library(opengv2::event ALIAS event)

target_include_directories(event PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/event>
        )

target_link_libraries(event PUBLIC
        frame
        )

# tool
add_executable(txt2bin ${CMAKE_CURRENT_SOURCE_DIR}/tool/txt2bin.cpp)
target_link_libraries(txt2bin event)

# test
add_executable(unit_test_event ${CMAKE_CURRENT_SOURCE_DIR}/test/unit_test.cpp)
target_link_libraries(unit_test_event event)