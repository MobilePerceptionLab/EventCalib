//
// Created by huangkun on 2020/10/8.
//

#ifndef OPENGV2_PARAMETERS_HPP
#define OPENGV2_PARAMETERS_HPP

#include <opencv2/opencv.hpp>

#include <memory>

struct CirclePatternParameters {
    typedef std::shared_ptr<CirclePatternParameters> Ptr;

    CirclePatternParameters(const cv::FileStorage &node) {
        node["BoardSize_Cols"] >> cols;
        node["BoardSize_Rows"] >> rows;
        node["Square_Size"] >> squareSize;
        node["Is_Pattern_Asymmetric"] >> isAsymmetric;
        node["Circles_Radius"] >> circleRadius;
    }

    bool isAsymmetric;
    int rows, cols;
    double circleRadius;
    double squareSize;
};

struct CalibrationSetting {
    typedef std::shared_ptr<CalibrationSetting> Ptr;

    CalibrationSetting(const cv::FileStorage &node) {
        circlePatternParameters = std::make_shared<CirclePatternParameters>(node);
        node["Calibrate_FixAspectRatio"] >> aspectRatio;
        node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
        node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
        node["Calibrate_UseFisheyeModel"] >> useFisheye;
        node["Fix_K1"] >> fixK1;
        node["Fix_K2"] >> fixK2;
        node["Fix_K3"] >> fixK3;
        node["Fix_K4"] >> fixK4;
        node["Fix_K5"] >> fixK5;
        node["Calibrate_NrOfFrameToUse"] >> NumOfFrameToUse;

        validate();
    }

    void validate() {
        flag = 0;
        if (calibFixPrincipalPoint) flag |= cv::CALIB_FIX_PRINCIPAL_POINT;
        if (calibZeroTangentDist) flag |= cv::CALIB_ZERO_TANGENT_DIST;
        if (aspectRatio) flag |= cv::CALIB_FIX_ASPECT_RATIO;
        if (fixK1) flag |= cv::CALIB_FIX_K1;
        if (fixK2) flag |= cv::CALIB_FIX_K2;
        if (fixK3) flag |= cv::CALIB_FIX_K3;
        if (fixK4) flag |= cv::CALIB_FIX_K4;
        if (fixK5) flag |= cv::CALIB_FIX_K5;
        flag |= cv::CALIB_FIX_K6;

        if (useFisheye) {
            // the fisheye model has its own enum, so overwrite the flags
            flag = cv::fisheye::CALIB_FIX_SKEW | cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
            if (fixK1) flag |= cv::fisheye::CALIB_FIX_K1;
            if (fixK2) flag |= cv::fisheye::CALIB_FIX_K2;
            if (fixK3) flag |= cv::fisheye::CALIB_FIX_K3;
            if (fixK4) flag |= cv::fisheye::CALIB_FIX_K4;
            if (calibFixPrincipalPoint) flag |= cv::fisheye::CALIB_FIX_PRINCIPAL_POINT;
        }
    }

    CirclePatternParameters::Ptr circlePatternParameters;
    int NumOfFrameToUse;
    float aspectRatio;           // The aspect ratio
    bool calibZeroTangentDist;   // Assume zero tangential distortion
    bool calibFixPrincipalPoint; // Fix the principal point at the center
    bool useFisheye;             // use fisheye camera model for calibration
    bool fixK1;                  // fix K1 distortion coefficient
    bool fixK2;                  // fix K2 distortion coefficient
    bool fixK3;                  // fix K3 distortion coefficient
    bool fixK4;                  // fix K4 distortion coefficient
    bool fixK5;                  // fix K5 distortion coefficient

    int flag;
};

#endif //OPENGV2_PARAMETERS_HPP
