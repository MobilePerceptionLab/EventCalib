//
// Created by huangkun on 2020/8/11.
//

#include <opengv2/sensor/OmniScaramuzzaCamera.hpp>

opengv2::OmniScaramuzzaCamera::OmniScaramuzzaCamera(const Eigen::Ref<const Eigen::Vector2d> &size,
                                                    const Eigen::Ref<const Eigen::VectorXd> &polynomial,
                                                    const Eigen::Ref<const Eigen::Vector2d> &principal_point,
                                                    const Eigen::Ref<const Eigen::Vector3d> &distortion,
                                                    const Eigen::Ref<const Eigen::VectorXd> &inverse_polynomial,
                                                    cv::Mat mask) :
        CameraBase(size, mask), polynomial_(polynomial), principal_point_(principal_point),
        inverse_polynomial_(inverse_polynomial),
        affine_correction_(distortionToAffineCorrection(distortion)),
        affine_correction_inverse_(affine_correction_.inverse()) {}

Eigen::Matrix2d
opengv2::OmniScaramuzzaCamera::distortionToAffineCorrection(const Eigen::Ref<const Eigen::Vector3d> &distortion) {
    Eigen::Matrix2d result;
    // [c, d; e, 1]
    result << distortion(0), distortion(1), distortion(2), 1;
    return result;
}

inline Eigen::Vector3d opengv2::OmniScaramuzzaCamera::invProject(const Eigen::Ref<const Eigen::Vector2d> &p) const {
    const Eigen::Vector2d rectified =
            affine_correction_inverse_ * (p - principal_point_);
    const double rho = rectified.norm();

    Eigen::Vector3d bv;
    bv.head<2>() = rectified;

    bv(2) = 0;
    for (int i = polynomial_.size() - 1; i >= 0; --i) {
        bv(2) = polynomial_(i) + bv(2) * rho;
    }

    bv.normalize();
    return bv;
}

inline Eigen::Vector2d opengv2::OmniScaramuzzaCamera::project(const Eigen::Ref<const Eigen::Vector3d> &Xc) const {
    const double x = Xc[0];
    const double y = Xc[1];
    const double z = Xc[2];
    const double xy_norm2 = std::pow(x, 2) + std::pow(y, 2);
    const double xy_norm = std::sqrt(xy_norm2);
    const double z_by_xy_norm = z / xy_norm;
    const double theta = std::atan(z_by_xy_norm);

    Eigen::VectorXd theta_powers(inverse_polynomial_.size());
    theta_powers[0] = 1.0;
    for (int i = 1; i < theta_powers.size(); ++i) {
        theta_powers[i] = theta_powers[i - 1] * theta;
    }

    const double rho = inverse_polynomial_.dot(theta_powers);

    Eigen::Vector2d raw_uv;
    raw_uv(0) = x / xy_norm * rho;
    raw_uv(1) = y / xy_norm * rho;

    return affine_correction_ * raw_uv + principal_point_;
}

Eigen::Vector2d opengv2::OmniScaramuzzaCamera::undistortPoint(const Eigen::Ref<const Eigen::Vector2d> &p) const {
    // TODO
}

cv::Mat opengv2::OmniScaramuzzaCamera::undistortImage(cv::Mat image) {
    // TODO: undistortImage
}

opengv2::OmniScaramuzzaCamera::OmniScaramuzzaCamera(const cv::FileNode &sensorNode) : CameraBase(sensorNode) {
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
    principal_point_ = Eigen::Vector2d(v.data());
    std::cout << "principal_point: " << principal_point_.transpose() << std::endl;

    v.clear();
    data = sensorNode["distortion"];
    if (!data.isSeq())
        throw std::invalid_argument(sensorNode.name() + ": distortion");
    for (cv::FileNodeIterator dataItr = data.begin(); dataItr != data.end(); dataItr++) {
        v.push_back(*dataItr);
    }
    if (v.size() != 3)
        throw std::invalid_argument(sensorNode.name() + ": distortion");
    Eigen::Vector3d distortion(v.data());
    std::cout << "distortion: " << distortion.transpose() << std::endl;
    affine_correction_ = distortionToAffineCorrection(distortion);
    affine_correction_inverse_ = affine_correction_.inverse();

    v.clear();
    data = sensorNode["poly"];
    if (!data.isSeq())
        throw std::invalid_argument(sensorNode.name() + ": poly");
    for (cv::FileNodeIterator dataItr = data.begin(); dataItr != data.end(); dataItr++) {
        v.push_back(*dataItr);
    }
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> poly_tmp(v.data(), v.size());
    polynomial_ = poly_tmp;
    std::cout << "poly: " << polynomial_.transpose() << std::endl;

    v.clear();
    data = sensorNode["inv_poly"];
    if (!data.isSeq())
        throw std::invalid_argument(sensorNode.name() + ": inv_poly");
    for (cv::FileNodeIterator dataItr = data.begin(); dataItr != data.end(); dataItr++) {
        v.push_back(*dataItr);
    }
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> inv_poly_tmp(v.data(), v.size());
    inverse_polynomial_ = inv_poly_tmp;
    std::cout << "inv_poly: " << inverse_polynomial_.transpose() << std::endl;
}