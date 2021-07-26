//
// Created by huangkun on 2020/10/21.
//
#include <iostream>
#include <opengv2/sensor/PinholeCamera.hpp>

using namespace opengv2;

int main(int argc, char **argv) {
    Eigen::Vector4d radial(-0.34991902, -0.014698517, 0.59684463, 0);
    Eigen::VectorXd inverse = PinholeCamera::inverseRadialDistortion(radial);

    Eigen::Matrix3d K;
    K << 359.67525, 0, 172.5,
            0, 359.67525, 129.5,
            0, 0, 1;

    Eigen::Vector3d p(50, 50, 1);
    Eigen::Vector3d Xc = K.inverse() * p;
    Xc /= Xc[2];

    // inverse
    double xx = Xc[0] * Xc[0];
    double yy = Xc[1] * Xc[1];
    double r2 = xx + yy;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    double r8 = r6 * r2;
    double r10 = r8 * r2;
    Eigen::VectorXd r_coeff(5);
    r_coeff << r2, r4, r6, r8, r10;
    Xc[0] *= 1 + r_coeff.transpose() * inverse;
    Xc[1] *= 1 + r_coeff.transpose() * inverse;

    // distortion
    xx = Xc[0] * Xc[0];
    yy = Xc[1] * Xc[1];
    r2 = xx + yy;
    r4 = r2 * r2;
    r6 = r4 * r2;
    r_coeff.resize(3);
    r_coeff << r2, r4, r6;
    Xc[0] *= 1 + r_coeff.transpose() * radial.block<3, 1>(0, 0);
    Xc[1] *= 1 + r_coeff.transpose() * radial.block<3, 1>(0, 0);

    Eigen::Vector3d p_new = K * Xc;
    p_new /= p_new[2];

    std::cout << (p_new - p).norm() << std::endl;

    return 0;
}