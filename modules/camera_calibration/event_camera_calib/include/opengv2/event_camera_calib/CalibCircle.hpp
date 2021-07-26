//
// Created by huangkun on 2020/10/7.
//

#ifndef OPENGV2_CALIBCIRCLE_HPP
#define OPENGV2_CALIBCIRCLE_HPP

namespace opengv2 {
    class CalibCircle : public FeatureBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CalibCircle(const Eigen::Ref<const Eigen::Vector2d> &center, double radius)
                : FeatureBase(center), radius(radius) {}

        double radius;
    };
}

#endif //OPENGV2_CALIBCIRCLE_HPP
