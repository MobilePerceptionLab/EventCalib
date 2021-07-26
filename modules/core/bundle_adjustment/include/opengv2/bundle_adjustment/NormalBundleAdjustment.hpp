//
// Created by huangkun on 2020/1/11.
//

#ifndef OPENGV2_NORMALBUNDLEADJUSTMENT_HPP
#define OPENGV2_NORMALBUNDLEADJUSTMENT_HPP

#include <Eigen/Eigen>

#include <opengv2/bundle_adjustment/BundleAdjustmentBase.hpp>

namespace opengv2 {
    class NormalBundleAdjustment : public BundleAdjustmentBase {
    public:
        NormalBundleAdjustment(bool fixTsb, bool robust, bool planarConstrain);

        void
        run(const std::map<double, Bodyframe::Ptr> &keyframes,
            const std::map<int, LandmarkBase::Ptr> &landmarks) override;

        struct ReprojectionError {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            explicit ReprojectionError(const Eigen::Ref<const Eigen::Vector3d> &bearingVector, bool isMono)
                    : isMono_(isMono), bearingVector_(bearingVector) {}

            template<typename T>
            bool operator()(const T *const Qbw_twb, const T *const lm, const T *const Tcb, T *residuals) const {
                Eigen::Quaternion<T> Qbw(Qbw_twb);
                Qbw.normalize();
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const twb(Qbw_twb + 4);
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const Xw(lm);
                Eigen::Quaternion<T> Qcb(Tcb);
                Qcb.normalize();
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const tcb(Tcb + 4);
                Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);

                Eigen::Matrix<T, 3, 1> Xb = Qbw * (Xw - twb);
                Eigen::Matrix<T, 3, 1> Xc = Qcb * Xb + tcb;
                if (isMono_)
                    Xc.normalize();

                res = bearingVector_ - Xc;
                return true;
            }

            static ceres::CostFunction *
            Create(const Eigen::Ref<const Eigen::Vector3d> &bearingVector, bool isMono) {
                return (new ceres::AutoDiffCostFunction<ReprojectionError, 3, 7, 3, 7>(
                        new ReprojectionError(bearingVector, isMono)));
            }

            bool isMono_;
            Eigen::Vector3d bearingVector_;
        };

    protected:
        void optimize(ceres::Problem &problem,
                      const std::map<double, Bodyframe::Ptr> &keyframes,
                      const std::map<int, LandmarkBase::Ptr> &landmarks) override;

        bool fixTsb_;
        bool robust_;
        bool planarConstrain_;
    };
}

#endif //OPENGV2_NORMALBUNDLEADJUSTMENT_HPP
