//
// Created by huangkun on 2020/7/22.
//

#ifndef OPENGV2_RELATIVECONSTRAINTBA_HPP
#define OPENGV2_RELATIVECONSTRAINTBA_HPP

#include <opengv2/bundle_adjustment/NormalBundleAdjustment.hpp>

namespace opengv2 {
    class RelativeConstraintBA : public NormalBundleAdjustment {
    public:
        RelativeConstraintBA(bool fixTsb, bool robust, bool planarConstrain, double weight);

        struct RelativeConstraint {
            template<typename T>
            bool operator()(const T *const Qb1w_twb1, const T *const Qb2w_twb2, T *residuals) const {
                Eigen::Quaternion <T> Qb1w(Qb1w_twb1);
                Qb1w.normalize();
                Eigen::Map < Eigen::Matrix < T, 3, 1 > const> const twb1(Qb1w_twb1
                +4);
                Eigen::Quaternion<T> Qb2w(Qb2w_twb2);
                Qb2w.normalize();
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const twb2(Qb2w_twb2 + 4);
                Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);

                Eigen::Quaternion<T> Qb1b2 = Qb1w * Qb2w.conjugate();
                Eigen::Matrix<T, 3, 1> tb1b2 = Qb1w * (twb2 - twb1);
                tb1b2.normalize();

                // (I+R)*[0;1;0]
                const T &Qx = Qb1b2.x();
                const T &Qy = Qb1b2.y();
                const T &Qz = Qb1b2.z();
                const T &Qw = Qb1b2.w();
                Eigen::Matrix<T, 3, 1> r(2. * (Qy * Qx - Qz * Qw), 1. - 2. * (Qx * Qx + Qz * Qz),
                                         2. * (Qz * Qy + Qx * Qw));
                r.normalize();
                r[1] += 1.;
                r.normalize();

                res = r.cross(tb1b2);
                return true;
            }

            static ceres::CostFunction *
            Create() {
                return (new ceres::AutoDiffCostFunction<RelativeConstraint, 3, 7, 7>(
                        new RelativeConstraint()));
            }
        };

    protected:
        void optimize(ceres::Problem &problem,
                      const std::map<double, Bodyframe::Ptr> &keyframes,
                      const std::map<int, LandmarkBase::Ptr> &landmarks) override;

        double weight_;
    };
}

#endif //OPENGV2_RELATIVECONSTRAINTBA_HPP
