//
// Created by huangkun on 2020/9/15.
//

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include <opengv2/event_camera_calib/EventCalibIni.hpp>
#include <opengv2/event/EventFrame.hpp>
#include <opengv2/system/SystemBase.hpp>
#include <opengv2/sensor/PinholeCamera.hpp>
#include <opengv2/feature/FeatureIdentifier.hpp>

opengv2::EventCalibIni::EventCalibIni(MapBase::Ptr map, CalibrationSetting::Ptr calibrationSetting,
                                      double motionTimeStep)
        : TrackingBase(std::move(map)), calibrationSetting_(calibrationSetting), motionTimeStep_(motionTimeStep) {}

bool opengv2::EventCalibIni::initialization(opengv2::Bodyframe::Ptr bf) {
    system_->map->addFrame(bf);
    return true;
}

bool opengv2::EventCalibIni::track(opengv2::Bodyframe::Ptr bf) {
    /****** check similarity with nearby pattern ******/
    Bodyframe::Ptr bf_ref;
    system_->map->keyframeLockShared();
    auto itr = system_->map->keyframes().lower_bound(bf->timeStamp());
    if (itr != system_->map->keyframes().end())
        bf_ref = itr->second;
    else {
        if (system_->map->keyframes().empty())
            throw std::logic_error("Tracking Not Initialized!");

        bf_ref = system_->map->keyframes().rbegin()->second;
    }
    system_->map->keyframeUnlockShared();

    double duration = std::abs(bf->timeStamp() - bf_ref->timeStamp());
    // identify outliers that pattern orientation changed in a very short time
    auto ref = dynamic_cast<EventFrame *>(bf_ref->frame(0).get());
    auto cur = dynamic_cast<EventFrame *>(bf->frame(0).get());

    const auto &height = calibrationSetting_->circlePatternParameters->rows;
    const auto &width = calibrationSetting_->circlePatternParameters->cols;
    vectorofEigenMatrix<Eigen::Vector2d> refLineSet, curLineSet;
    for (int i = 0; i < height; i++) {
        Eigen::MatrixXd A(width, 3);
        A.col(2).setOnes();
        for (int j = 0; j < width; j++) {
            A.block<1, 2>(j, 0) = ref->features()[i * width + j]->location();
        }

        Eigen::JacobiSVD svd(A, Eigen::ComputeThinV);
        refLineSet.emplace_back(svd.matrixV()(1, 2), -svd.matrixV()(0, 2)); // B -A

        Eigen::Vector2d d = A.block<1, 2>(width - 1, 0) - A.block<1, 2>(0, 0);// B -A or -B A
        if (refLineSet.back().dot(d) < 0)
            refLineSet.back() = -refLineSet.back();
    }
    for (int i = 0; i < height; i++) {
        Eigen::MatrixXd A(width, 3);
        A.col(2).setOnes();
        for (int j = 0; j < width; j++) {
            A.block<1, 2>(j, 0) = cur->features()[i * width + j]->location();
        }

        Eigen::JacobiSVD svd(A, Eigen::ComputeThinV);
        curLineSet.emplace_back(svd.matrixV()(1, 2), -svd.matrixV()(0, 2)); // B -A

        Eigen::Vector2d d = A.block<1, 2>(width - 1, 0) - A.block<1, 2>(0, 0);// B -A or -B A
        if (curLineSet.back().dot(d) < 0)
            curLineSet.back() = -curLineSet.back();
    }

    std::vector<double> thetaSet;
    for (int i = 0; i < height; ++i) {
        double theta = std::acos(refLineSet[i].dot(curLineSet[i]) / (refLineSet[i].norm() * curLineSet[i].norm()));
        thetaSet.push_back(theta);
    }
    std::nth_element(thetaSet.begin(), thetaSet.begin() + thetaSet.size() / 2, thetaSet.end());

    if (thetaSet[thetaSet.size() / 2] / duration < (5e-4 * M_PI) / motionTimeStep_) { // Angular velocity rad/second
        system_->map->addFrame(bf);

        /*if (thetaSet[thetaSet.size() / 2] > M_PI / 4) {
            std::cerr << "Please check!" << std::endl;
            cv::waitKey();
        }*/

        if (system_->viewer != nullptr)
            system_->viewer->updateViewer();

        return true;
    }

    return false;
}

void opengv2::EventCalibIni::calcBoardCornerPositions(std::vector<cv::Point3f> &corners) {
    corners.clear();

    if (calibrationSetting_->circlePatternParameters->isAsymmetric) {
        for (int i = 0; i < calibrationSetting_->circlePatternParameters->rows; i++)
            for (int j = 0; j < calibrationSetting_->circlePatternParameters->cols; j++)
                corners.emplace_back((2 * j + i % 2) * calibrationSetting_->circlePatternParameters->squareSize,
                                     i * calibrationSetting_->circlePatternParameters->squareSize, 0);
    } else {
        for (int i = 0; i < calibrationSetting_->circlePatternParameters->rows; ++i)
            for (int j = 0; j < calibrationSetting_->circlePatternParameters->cols; ++j)
                corners.emplace_back(j * calibrationSetting_->circlePatternParameters->squareSize,
                                     i * calibrationSetting_->circlePatternParameters->squareSize, 0);
    }
}

static double computeReprojectionErrors(const std::vector<std::vector<cv::Point3f> > &objectPoints,
                                        const std::vector<std::vector<cv::Point2f> > &imagePoints,
                                        const std::vector<cv::Mat> &rvecs, const std::vector<cv::Mat> &tvecs,
                                        const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                                        std::vector<float> &perViewErrors, bool fisheye) {
    std::vector<cv::Point2f> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for (size_t i = 0; i < objectPoints.size(); ++i) {
        if (fisheye) {
            cv::fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix,
                                       distCoeffs);
        } else {
            projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
        }
        err = norm(imagePoints[i], imagePoints2, cv::NORM_L2);

        size_t n = objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}

bool opengv2::EventCalibIni::cvCalibration() {
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<float> reprojErrs;

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    if (calibrationSetting_->aspectRatio != 0)
        cameraMatrix.at<double>(0, 0) = calibrationSetting_->aspectRatio;

    cv::Mat distCoeffs;
    if (calibrationSetting_->useFisheye) {
        distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
    } else {
        distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
    }

    int step = system_->map->frameNum() / calibrationSetting_->NumOfFrameToUse;
    if (step == 0) {
        calibrationSetting_->NumOfFrameToUse = system_->map->frameNum();
        step = 1;
    }

    // select frames for intrinsic initialization
    std::vector<std::vector<cv::Point2f>> imagePoints(calibrationSetting_->NumOfFrameToUse);
    int counter = 0;
    system_->map->keyframeLockShared();
    for (auto itr = system_->map->keyframes().cbegin();
         counter < calibrationSetting_->NumOfFrameToUse; std::advance(itr, step), counter++) {
        auto bf = itr->second;
        auto cf = dynamic_cast<EventFrame *>(bf->frame(0).get());
        for (const auto &f:cf->features()) {
            imagePoints[counter].emplace_back(f->location()[0], f->location()[1]);
        }
    }
    system_->map->keyframeUnlockShared();

    std::vector<std::vector<cv::Point3f> > objectPoints(1);
    calcBoardCornerPositions(objectPoints[0]);
    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    //Find intrinsic and extrinsic camera parameters
    double rms;
    auto camera = dynamic_cast<PinholeCamera *>(system_->map->firstKeyframe()->frame(0)->sensor().get());
    cv::Size imageSize(camera->size()[0], camera->size()[1]);
    if (calibrationSetting_->useFisheye) {
        cv::Mat _rvecs, _tvecs;
        rms = cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, _rvecs,
                                     _tvecs, calibrationSetting_->flag);

        rvecs.reserve(_rvecs.rows);
        tvecs.reserve(_tvecs.rows);
        for (int i = 0; i < int(objectPoints.size()); i++) {
            rvecs.push_back(_rvecs.row(i));
            tvecs.push_back(_tvecs.row(i));
        }
    } else {
        rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs,
                              calibrationSetting_->flag | cv::CALIB_USE_LU);
    }

    std::cout << "Re-projection error reported by calibrateCamera: " << rms << std::endl;

    bool ok = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);
    double totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
                                                   distCoeffs, reprojErrs, calibrationSetting_->useFisheye);

    std::cout << (ok ? "Calibration succeeded" : "Calibration failed")
              << ". avg re projection error = " << totalAvgErr << std::endl;

    int counter1 = 0, counter2 = 0, counter3 = 0;
    if (ok) {
        Eigen::Matrix3d K;
        cv::cv2eigen(cameraMatrix, K);
        Eigen::VectorXd distortion;
        cv::cv2eigen(distCoeffs, distortion);
        std::cout << K << std::endl;
        std::cout << distortion.transpose() << std::endl;

        // set camera intrinsics
        camera->setK(K);
        camera->distCoeffs() = distortion;

        std::map<double, Bodyframe::Ptr> buffer = map_->copyKeyframes();

        // add pattern landmarks to map
        for (int i = 0; i < objectPoints[0].size(); ++i) {
            Eigen::Vector3d Xw(objectPoints[0][i].x, objectPoints[0][i].y, objectPoints[0][i].z);
            LandmarkBase::Ptr lm = std::make_shared<LandmarkBase>(i, Xw);
            for (auto &itr : buffer) {
                auto &bf = itr.second;
                auto cf = dynamic_cast<EventFrame *>(bf->frame(0).get());
                if (i < cf->features().size()) {
                    FeatureIdentifier fi(bf->timeStamp(), 0, i, cf->features()[i]);
                    lm->addObservation(fi);
                    cf->features()[i]->setLandmark(lm);
                }
            }
            system_->map->addLandmark(lm);
        }

        // clear frames in map, lm->observations() will be cleared.
        for (auto &itr : buffer) {
            auto &bf = itr.second;
            system_->map->removeFrame(bf->timeStamp());
        }

        // solvePnPRansac: initialize frame poses
        for (auto &itr : buffer) {
            auto &bf = itr.second;
            cv::Mat rvec, tvec;
            std::vector<cv::Point2f> imageP;
            std::vector<int> inliers;
            auto cf = dynamic_cast<EventFrame *>(bf->frame(0).get());
            for (const auto &f:cf->features()) {
                imageP.emplace_back(f->location()[0], f->location()[1]);
            }
            cv::solvePnPRansac(objectPoints[0], imageP, cameraMatrix, distCoeffs, rvec, tvec, false, 50, 4.0, 0.99,
                               inliers, cv::SOLVEPNP_IPPE);

            /*// check bad frame
            if (inliers.size() < imageP.size() * 0.25) {
                counter3++;
                continue;
            }*/

            cv::Mat cvRsw;
            cv::Rodrigues(rvec, cvRsw);
            Eigen::Matrix3d Rsw;
            Eigen::Vector3d tsw;
            cv::cv2eigen(cvRsw, Rsw);
            cv::cv2eigen(tvec, tsw);
            Eigen::Quaterniond Qsw(Rsw);
            Qsw.normalize();

            // set poses
            Eigen::Vector3d twb = Qsw.conjugate() * (Bodyframe::tsb(0) - tsw);
            Eigen::Quaterniond unitQwb = Qsw.conjugate() * Bodyframe::unitQsb(0);
            bf->setPose(twb, unitQwb);


            // check pose in map
            if (checkPose(bf)) {
                std::unordered_set<int> outlierIdxs;
                outlierIdxs.reserve(imageP.size());
                for (int i = 0; i < imageP.size(); ++i) {
                    outlierIdxs.insert(i);
                }
                for (auto inlier: inliers) {
                    outlierIdxs.erase(inlier);
                }

                // refinement on outlier features
                if (cf->rectifyFeatures(outlierIdxs, Rsw, tsw)) {
                    system_->map->addFrame(bf);
                } else {
                    counter2++;
                }
            } else {
                counter1++;
            }
        }

        std::map<double, Bodyframe::Ptr> frames = map_->copyKeyframes();
        for (auto &itr : frames) {
            auto &bf = itr.second;
            for (int i = 0; i < bf->size(); ++i) {
                if (auto cf = dynamic_cast<CameraFrame *>(bf->frame(i).get())) {
                    for (int j = 0; j < cf->features().size(); j++) {
                        auto f = cf->features()[j];
                        if (auto lm = f->landmark()) {
                            FeatureIdentifier fi(bf->timeStamp(), i, j, f);
                            lm->addObservation(fi);
                        }
                    }
                }
            }
        }

        system_->viewer->updateViewer();
    }

    std::cout << counter1 << " frames discard by checkPose." << std::endl;
    std::cout << counter2 << " frames discard by rectifyFeature." << std::endl;
    //std::cout << counter3 << " frames discard by check inlierNum." << std::endl;
    return ok;
}

bool opengv2::EventCalibIni::checkPose(Bodyframe::Ptr cur) const {
    if (system_->map->empty()) {
        return true;
    }
    auto ref = system_->map->lastKeyframe();

    double duration = cur->timeStamp() - ref->timeStamp();

    double v_t = (ref->unitQwb().conjugate() * (cur->twb() - ref->twb())).norm() / duration;
    double v_R = std::abs(cur->unitQwb().angularDistance(ref->unitQwb()) / duration);

    //std::cout << "v_t: " << v_t << " v_R: " << v_R << std::endl;

    if (v_t < (2.5e-1 / motionTimeStep_) * 2 /*cm per second*/ &&
        v_R < (5e-4 * M_PI) * 2 / motionTimeStep_ /*rad per second*/)
        return true;
    else
        return false;
}
