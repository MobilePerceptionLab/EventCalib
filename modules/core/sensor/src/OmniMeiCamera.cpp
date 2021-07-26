//
// Created by huangkun on 2021/1/6.
//

#include <opengv2/sensor/OmniMeiCamera.hpp>

opengv2::OmniMeiCamera::OmniMeiCamera(const Eigen::Ref<const Eigen::Vector2d> &size,
                                      const Eigen::Ref<const Eigen::VectorXd> &intrinsic,
                                      const Eigen::Ref<const Eigen::VectorXd> &distort) :
        CameraBase(size), intrinsic_(intrinsic), distort_(distort) {}

Eigen::Vector2d opengv2::OmniMeiCamera::project(const Eigen::Ref<const Eigen::Vector3d> &Xc) const {
    double xi = intrinsic_[0];
    double fu = intrinsic_[1];
    double fv = intrinsic_[2];
    double cu = intrinsic_[3];
    double cv = intrinsic_[4];

    double fov_parameter = (xi <= 1.0) ? xi : 1.0 / xi;

    Eigen::Vector2d outKeypoint;

    double d = Xc.norm();

    // Check if point will lead to a valid projection
    if (Xc[2] <= -(fov_parameter * d))
        return outKeypoint;

    double rz = 1.0 / (Xc[2] + xi * d);
    outKeypoint[0] = Xc[0] * rz;
    outKeypoint[1] = Xc[1] * rz;

    outKeypoint = distort(outKeypoint);

    outKeypoint[0] = fu * outKeypoint[0] + cu;
    outKeypoint[1] = fv * outKeypoint[1] + cv;
    return outKeypoint;
}

Eigen::Vector3d opengv2::OmniMeiCamera::invProject(const Eigen::Ref<const Eigen::Vector2d> &p) const {
    double xi = intrinsic_[0];
    double fu = intrinsic_[1];
    double fv = intrinsic_[2];
    double cu = intrinsic_[3];
    double cv = intrinsic_[4];

    double recip_fu = 1.0 / fu;
    double recip_fv = 1.0 / fv;

    Eigen::Vector3d outPoint;
    // Unproject...
    outPoint[0] = recip_fu * (p[0] - cu);
    outPoint[1] = recip_fv * (p[1] - cv);

    // Re-distort
    Eigen::Vector2d temp = outPoint.block<2, 1>(0, 0);
    temp = undistortPoint(temp);
    outPoint.block<2, 1>(0, 0) = temp;

    double rho2_d = outPoint[0] * outPoint[0] + outPoint[1] * outPoint[1];
    outPoint[2] = 1.0 - xi * (rho2_d + 1.0) / (xi + sqrt(1.0 + (1.0 - xi * xi) * rho2_d));

    return outPoint;
}

Eigen::Vector2d opengv2::OmniMeiCamera::undistortPoint(const Eigen::Ref<const Eigen::Vector2d> &p) const {
    Eigen::Vector2d ybar = p;
    const int n = 5;
    Eigen::Matrix2d F;

    Eigen::Vector2d y_tmp;
    for (int i = 0; i < n; i++) {
        y_tmp = ybar;
        y_tmp = distort(y_tmp, F);

        Eigen::Vector2d e(p - y_tmp);
        Eigen::Vector2d du = (F.transpose() * F).inverse() * F.transpose() * e;
        ybar += du;
        if (e.dot(e) < 1e-15)
            break;
    }

    return ybar;
}

cv::Mat opengv2::OmniMeiCamera::undistortImage(cv::Mat image) {

}

Eigen::Vector2d opengv2::OmniMeiCamera::distort(const Eigen::Ref<const Eigen::Vector2d> &pt) const {
    double k1 = distort_[0];
    double k2 = distort_[1];
    double p1 = distort_[2];
    double p2 = distort_[3];

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = pt[0] * pt[0];
    my2_u = pt[1] * pt[1];
    mxy_u = pt[0] * pt[1];
    rho2_u = mx2_u + my2_u;

    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;

    Eigen::Vector2d uv;
    uv[0] = pt[0] + pt[0] * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u);
    uv[1] = pt[1] + pt[1] * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);

    return uv;
}

Eigen::Vector2d opengv2::OmniMeiCamera::distort(const Eigen::Ref<const Eigen::Vector2d> &pt,
                                                Eigen::Ref <Eigen::Matrix2d> J) const {
    double k1 = distort_[0];
    double k2 = distort_[1];
    double p1 = distort_[2];
    double p2 = distort_[3];

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;
    J.setZero();
    mx2_u = pt[0] * pt[0];
    my2_u = pt[1] * pt[1];
    mxy_u = pt[0] * pt[1];
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    J(0, 0) = 1.0 + rad_dist_u + k1 * 2.0 * mx2_u + k2 * rho2_u * 4.0 * mx2_u
              + 2.0 * p1 * pt[1] + 6.0 * p2 * pt[0];
    J(1, 0) = k1 * 2.0 * pt[0] * pt[1] + k2 * 4.0 * rho2_u * pt[0] * pt[1]
              + p1 * 2.0 * pt[0] + 2.0 * p2 * pt[1];
    J(0, 1) = J(1, 0);
    J(1, 1) = 1.0 + rad_dist_u + k1 * 2.0 * my2_u + k2 * rho2_u * 4.0 * my2_u
              + 6.0 * p1 * pt[1] + 2.0 * p2 * pt[0];

    Eigen::Vector2d uv;
    uv[0] = pt[0] + pt[0] * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u);
    uv[1] = pt[1] + pt[1] * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);

    return uv;
}