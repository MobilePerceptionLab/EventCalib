//
// Created by huangkun on 2020/9/18.
//

#ifndef OPENGV2_EVENTFRAME_HPP
#define OPENGV2_EVENTFRAME_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <unordered_set>

#include <opengv2/frame/CameraFrame.hpp>
#include <opengv2/sensor/CameraBase.hpp>
#include <opengv2/feature/FeatureBase.hpp>
#include <opengv2/event/Event.hpp>
#include <opengv2/event/EventContainer.hpp>
#include <opengv2/utility/utility.hpp>
#include <opengv2/frame/Bodyframe.hpp>

namespace opengv2 {
    class EventFrame : public CameraFrame {
    public:
        /**
         * @brief The duration should be in 3e-3 second order for hand-hold, smaller for fast movement.
         * Range: [duration.first, duration.second)
         */
        EventFrame(EventContainer::Ptr container, const std::pair<double, double> &duration);

        /**
         * @brief release event set to save space after feature extracted
         */
        inline void releaseEventSet() {
            positiveEvents_.clear();
            negativeEvents_.clear();
        }

        /*
        * rectify features after opencv calibration, rediscovery raw data
        */
        virtual bool rectifyFeatures(const std::unordered_set<int> &outlierIdxs,
                                     const Eigen::Ref<const Eigen::Matrix3d> &Rcw,
                                     const Eigen::Ref<const Eigen::Vector3d> &tcw) = 0;

        inline int eventsNum() const noexcept {
            return positiveEvents_.size() + negativeEvents_.size();
        }

        cv::Mat undistortedImage(CameraBase::Ptr camera) const;

    protected:
        EventContainer::Ptr container_;
        std::pair<double, double> duration_;

        // could be released after rectifying features.
        vectorofEigenMatrix<Eigen::Vector2d> positiveEvents_, negativeEvents_;
    };
}

#endif //OPENGV2_EVENTFRAME_HPP
