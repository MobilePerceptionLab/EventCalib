//
// Created by huangkun on 2020/10/7.
//

#ifndef OPENGV2_EVENTCALIBINI_HPP
#define OPENGV2_EVENTCALIBINI_HPP

#include <opengv2/tracking/TrackingBase.hpp>
#include <opengv2/frame/Bodyframe.hpp>
#include <opengv2/event_camera_calib/parameters.hpp>

namespace opengv2 {
    class EventCalibIni : public TrackingBase {
    public:
        explicit EventCalibIni(MapBase::Ptr map, CalibrationSetting::Ptr calibrationSetting, double motionTimeStep);

        /*
         * Initialization: using opencv function to initialize the pose and K;
         */
        bool cvCalibration();

        /*
         * identify outlier frames by velocity and angular velocity check
         */
        bool checkPose(Bodyframe::Ptr cur) const;

    protected:
        bool track(Bodyframe::Ptr bf) override;

        bool initialization(Bodyframe::Ptr bf) override;

        void calcBoardCornerPositions(std::vector<cv::Point3f> &corners);

        CalibrationSetting::Ptr calibrationSetting_;

        double motionTimeStep_;
    };
}

#endif //OPENGV2_EVENTCALIBINI_HPP
