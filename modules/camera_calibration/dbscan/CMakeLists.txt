# dbscan clustering, TODO: use own version with nanoflann
add_library(dbscan SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kdtree.cpp
        )
add_library(opengv2::dbscan ALIAS dbscan)

target_include_directories(dbscan PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/opengv2/dbscan>
        )