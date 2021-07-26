//
// Created by huangkun on 2020/8/11.
//

#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <nanoflann.hpp>
#include <KDTreeVectorOfVectorsAdaptor.h>

#include <opengv2/sensor/PinholeCamera.hpp>
#include <opengv2/utility/utility.hpp>

opengv2::PinholeCamera::PinholeCamera(const Eigen::Ref<const Eigen::Vector2d> &size,
                                      const Eigen::Ref<const Eigen::Matrix3d> &K,
                                      const Eigen::Ref<const Eigen::VectorXd> &distCoeffs,
                                      cv::Mat mask) :
        CameraBase(size, mask), K_(K), invK_(K.inverse()), distCoeffs_(distCoeffs),
        inverseRadialPoly_(Eigen::VectorXd()) {}

inline Eigen::Vector2d opengv2::PinholeCamera::project(const Eigen::Ref<const Eigen::Vector3d> &Xc) const {
    if (distCoeffs_.size() == 0) {
        Eigen::Vector3d p = K_ * Xc;
        p /= p(2);
        return p.block<2, 1>(0, 0);
    } else {
        // TODO: implement using Eigen
        std::vector<cv::Point3f> objectPoints(1, cv::Point3f(Xc[0], Xc[1], Xc[2]));
        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32F), rvec, distCoeffs, cameraMatrix;
        cv::Rodrigues(cv::Mat::eye(3, 3, CV_32F), rvec);
        cv::eigen2cv(distCoeffs_, distCoeffs);
        cv::eigen2cv(K_, cameraMatrix);
        std::vector<cv::Point2f> imagePoints;
        cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
        return Eigen::Vector2d(imagePoints[0].x, imagePoints[0].y);
    }
}

inline Eigen::Vector3d opengv2::PinholeCamera::invProject(const Eigen::Ref<const Eigen::Vector2d> &p) const {
    Eigen::Vector3d Xc;
    if (inverseRadialPoly_.size() != 0) {
        Xc = invK_ * Eigen::Vector3d(p(0), p(1), 1);
        Xc /= Xc[2];

        Eigen::VectorXd r_coeff(inverseRadialPoly_.size());
        r_coeff[0] = Xc[0] * Xc[0] + Xc[1] * Xc[1];
        for (int i = 1; i < inverseRadialPoly_.size(); ++i) {
            r_coeff[i] = r_coeff[i - 1] * r_coeff[0];
        }
        Xc[0] *= 1 + r_coeff.transpose() * inverseRadialPoly_;
        Xc[1] *= 1 + r_coeff.transpose() * inverseRadialPoly_;
    } else if (distCoeffs_.size() != 0) {
        // TODO: implement using Eigen
        std::vector<cv::Point2f> src, dst;
        src.emplace_back(p[0], p[1]);
        cv::Mat cameraMatrix, distCoeffs;
        cv::eigen2cv(K_, cameraMatrix);
        cv::eigen2cv(distCoeffs_, distCoeffs);
        cv::undistortPoints(src, dst, cameraMatrix, distCoeffs);
        Xc = Eigen::Vector3d(dst[0].x, dst[0].y, 1);
    } else {
        Xc = invK_ * Eigen::Vector3d(p(0), p(1), 1);
    }

    Xc.normalize();
    return Xc;
}

Eigen::VectorXd
opengv2::PinholeCamera::inverseRadialDistortion(const Eigen::Ref<const Eigen::Vector4d> &radialDistortion) {
    const Eigen::Ref<const Eigen::VectorXd> &k = radialDistortion;
    Eigen::VectorXd b(5);

    double k00 = k[0] * k[0];
    double k000 = k[0] * k00;
    double k0000 = k[0] * k000;
    double k00000 = k[0] * k0000;
    double k01 = k[0] * k[1];
    double k001 = k[0] * k01;
    double k0001 = k[0] * k001;
    double k11 = k[1] * k[1];
    double k011 = k[0] * k11;
    double k02 = k[0] * k[2];
    double k002 = k[0] * k02;
    double k12 = k[1] * k[2];
    double k03 = k[0] * k[3];

    b[0] = -k[0];
    b[1] = 3 * k00 - k[1];
    b[2] = -12 * k000 + 8 * k01 - k[2];
    b[3] = 55 * k0000 - 55 * k001 + 5 * k11 + 10 * k02 - k[3];
    b[4] = -273 * k00000 + 364 * k0001 - 78 * k011 - 78 * k002 + 12 * k12 + 12 * k03;

    return b;
}

Eigen::Vector2d opengv2::PinholeCamera::undistortPoint(const Eigen::Ref<const Eigen::Vector2d> &p) const {
    Eigen::Vector3d Xc;
    if (inverseRadialPoly_.size() != 0) {
        Xc = invK_ * Eigen::Vector3d(p(0), p(1), 1);
        Xc /= Xc[2];

        Eigen::VectorXd r_coeff(inverseRadialPoly_.size());
        r_coeff[0] = Xc[0] * Xc[0] + Xc[1] * Xc[1];
        for (int i = 1; i < inverseRadialPoly_.size(); ++i) {
            r_coeff[i] = r_coeff[i - 1] * r_coeff[0];
        }
        Xc[0] *= 1 + r_coeff.transpose() * inverseRadialPoly_;
        Xc[1] *= 1 + r_coeff.transpose() * inverseRadialPoly_;
    } else if (distCoeffs_.size() != 0) {
        // TODO: implement using Eigen
        std::vector<cv::Point2f> src, dst;
        src.emplace_back(p[0], p[1]);
        cv::Mat cameraMatrix, distCoeffs;
        cv::eigen2cv(K_, cameraMatrix);
        cv::eigen2cv(distCoeffs_, distCoeffs);
        cv::undistortPoints(src, dst, cameraMatrix, distCoeffs);
        Xc = Eigen::Vector3d(dst[0].x, dst[0].y, 1);
    } else {
        return p;
    }

    Eigen::Vector3d p_c = K_ * Xc;
    p_c /= p_c[2];
    return p_c.block<2, 1>(0, 0);
}

cv::Mat opengv2::PinholeCamera::undistortImage(cv::Mat src) {
    if (inverseRadialPoly_.size() != 0) {
        if (undistortMap_.empty()) {
            // Initialization
            int dstRows = std::round(size_[1] * 1.2);
            int dstCols = std::round(size_[0] * 1.2);
            int emptyEdge_r = std::round(size_[1] * 0.1);
            int emptyEdge_c = std::round(size_[0] * 0.1);
            undistortMap_.assign(dstRows, std::vector(dstCols, std::pair(std::vector<std::pair<int, int>>(),
                                                                         std::vector<double>())));

            vectorofEigenMatrix<Eigen::Vector2d> correctedSet;
            correctedSet.reserve(size_[0] * size_[1]);
            const int step = size_[0]; // width
            for (int idx = 0; idx < size_[0] * size_[1]; ++idx) { // index of cv::Mat, row major
                int row = idx / step; // y
                int col = idx % step; // x
                correctedSet.push_back(
                        undistortPoint(Eigen::Vector2d(col, row)) + Eigen::Vector2d(emptyEdge_c, emptyEdge_r));
            }

            KDTreeVectorOfVectorsAdaptor<vectorofEigenMatrix<Eigen::Vector2d>, double, 2, nanoflann::metric_L2_Simple>
                    kdTree(2, correctedSet, 10);
            std::vector<std::pair<size_t, double>> indicesDists;
            for (int i = 0; i < dstRows; ++i) { // y, height
                for (int j = 0; j < dstCols; ++j) { // x, width
                    Eigen::Vector2d p(j, i);
                    kdTree.index->radiusSearch(p.data(), 2 * 2 /*square pixel unit*/,
                                               indicesDists, nanoflann::SearchParams(32, 0, false));
                    for (const auto &pair: indicesDists) {
                        int idx = pair.first;
                        double d = pair.second;
                        int row = idx / step;
                        int col = idx % step;
                        if (d < 100 * std::numeric_limits<double>::epsilon()) {
                            d = 100 * std::numeric_limits<double>::epsilon();
                        }
                        undistortMap_[i][j].first.emplace_back(row, col);
                        undistortMap_[i][j].second.push_back(1 / d);
                    }
                }
            }
        }

        if (src.type() != CV_8UC3) {
            throw std::logic_error("only support CV_8UC3 for now.");
        }
        cv::Mat dst(undistortMap_.size(), undistortMap_[0].size(), CV_32FC3, cv::Vec3f(0, 0, 0));
        for (int i = 0; i < dst.rows; ++i) {
            for (int j = 0; j < dst.cols; ++j) {
                int len = undistortMap_[i][j].first.size();
                double w_sum = 0;
                for (int k = 0; k < len; ++k) {
                    int row = undistortMap_[i][j].first[k].first;
                    int col = undistortMap_[i][j].first[k].second;
                    double w = undistortMap_[i][j].second[k];
                    // convert to float, since opencv didn't do that(uchar * double, return uchar).
                    cv::Vec3f temp = src.at<cv::Vec3b>(row, col);
                    dst.at<cv::Vec3f>(i, j) += w * temp;
                    w_sum += w;
                }
                if (len != 0) {
                    dst.at<cv::Vec3f>(i, j) /= w_sum;
                }
            }
        }

        cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
        return dst;
    } else if (distCoeffs_.size() != 0) {
        if (map1_.empty()) {
            cv::Size imageSize(size_[0], size_[1]);
            cv::Mat cameraMatrix, distCoeffs;
            cv::eigen2cv(K_, cameraMatrix);
            cv::eigen2cv(distCoeffs_, distCoeffs);
            cv::initUndistortRectifyMap(
                    cameraMatrix, distCoeffs, cv::Mat(),
                    getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                    imageSize, CV_32FC1, map1_, map2_);
        }
        cv::Mat dst;
        remap(src, dst, map1_, map2_, cv::INTER_LINEAR);
        return dst;
    } else {
        return src;
    }
}

opengv2::PinholeCamera::PinholeCamera(const cv::FileNode &sensorNode) : CameraBase(sensorNode) {
    std::vector<double> v;
    cv::FileNode data;

    data = sensorNode["principal_point"];
    if (!data.isSeq())
        throw std::invalid_argument(sensorNode.name() + ": principal_point");
    for (cv::FileNodeIterator dataItr = data.begin(); dataItr != data.end(); dataItr++) {
        v.push_back(*dataItr);
    }
    if (v.size() != 2)
        throw std::invalid_argument(sensorNode.name() + ": principal_point");
    Eigen::Vector2d principal_point(v.data());
    std::cout << "principal_point: " << principal_point.transpose() << std::endl;

    v.clear();
    data = sensorNode["distortion_type"];
    if (!data.isString())
        throw std::invalid_argument(sensorNode.name() + ": distortion_type");
    std::string distortion_type = data.string();
    std::cout << "distortion_type: " << distortion_type << std::endl;

    v.clear();
    data = sensorNode["distortion"];
    if (!data.isSeq())
        throw std::invalid_argument(sensorNode.name() + ": distortion");
    for (cv::FileNodeIterator dataItr = data.begin(); dataItr != data.end(); dataItr++) {
        v.push_back(*dataItr);
    }
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> distortion_tmp(v.data(), v.size());
    Eigen::VectorXd distortion(distortion_tmp);
    std::cout << "distortion: " << distortion.transpose() << std::endl;

    v.clear();
    data = sensorNode["focal_length"];
    if (!data.isSeq())
        throw std::invalid_argument(sensorNode.name() + ": focal_length");
    for (cv::FileNodeIterator dataItr = data.begin(); dataItr != data.end(); dataItr++) {
        v.push_back(*dataItr);
    }
    if (v.size() != 2)
        throw std::invalid_argument(sensorNode.name() + ": focal_length");
    Eigen::Vector2d focal_length(v.data());
    std::cout << "focal_length: " << focal_length.transpose() << std::endl;

    K_ << focal_length[0], 0, principal_point[0], 0, focal_length[1], principal_point[1], 0, 0, 1;
    invK_ = K_.inverse();
    if (distortion_type == "OpenCV") {
        distCoeffs_ = distortion;
        inverseRadialPoly_ = Eigen::VectorXd();
    } else {
        distCoeffs_ = Eigen::VectorXd();
        inverseRadialPoly_ = distortion;
    }
}