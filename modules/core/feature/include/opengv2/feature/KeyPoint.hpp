//
// Created by huangkun on 2019/12/30.
//

#ifndef OPENGV2_KEYPOINT_HPP
#define OPENGV2_KEYPOINT_HPP

#include <opencv2/opencv.hpp>

#include <opengv2/feature/FeatureBase.hpp>

namespace opengv2 {
    class KeyPoint : public FeatureBase {
    public:
        explicit KeyPoint(const cv::KeyPoint &cvKeyPoint) :
                FeatureBase(Eigen::Vector2d(cvKeyPoint.pt.x, cvKeyPoint.pt.y)), cvKeyPoint(cvKeyPoint) {}

        inline void setLocation(const Eigen::Ref<const Eigen::VectorXd> &loc) noexcept override {
            loc_ = loc;
            cvKeyPoint.pt.x = loc[0];
            cvKeyPoint.pt.y = loc[1];
        }

        cv::KeyPoint cvKeyPoint;
    };
}

#endif //OPENGV2_KEYPOINT_HPP
