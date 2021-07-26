//
// Created by huangkun on 2020/9/28.
//

#ifndef OPENGV2_CV_CALIB_HPP
#define OPENGV2_CV_CALIB_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

namespace cv {
    /*
     * The function attempts to determine whether the input image contains a grid of circles.
     * If it is, the function locates centers of the circles. The function returns a non-zero value
     * if all of the centers have been found and they have been placed in a certain order
     * (row by row, left to right in every row). Otherwise,
     * if the function fails to find all the corners or reorder them, it returns 0.
     */
    bool findCirclesGrid(const std::vector<Point2f> &points_, Size patternSize,
                         OutputArray _centers, int flags,
                         const CirclesGridFinderParameters &parameters_ = CirclesGridFinderParameters());
}

#endif //OPENGV2_CV_CALIB_HPP
