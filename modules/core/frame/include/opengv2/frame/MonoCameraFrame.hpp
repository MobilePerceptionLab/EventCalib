//
// Created by huangkun on 2020/1/12.
//

#ifndef OPENGV2_MONOCAMERAFRAME_HPP
#define OPENGV2_MONOCAMERAFRAME_HPP

#include <opengv2/frame/CameraFrame.hpp>

namespace opengv2 {
    class MonoCameraFrame : public CameraFrame {
    public:
        MonoCameraFrame(cv::Mat image, const CameraBase::Ptr &camera) : CameraFrame(image, camera) {}

        std::vector <cv::Point2f> kpSet; // for cv::calcOpticalFlowPyrLK()
    };
}

#endif //OPENGV2_MONOCAMERAFRAME_HPP
