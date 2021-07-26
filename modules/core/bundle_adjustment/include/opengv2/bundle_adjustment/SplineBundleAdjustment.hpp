//
// Created by huangkun on 2019/12/30.
//

#ifndef OPENGV2_SPLINEBUNDLEADJUSTMENT_HPP
#define OPENGV2_SPLINEBUNDLEADJUSTMENT_HPP

#include <Eigen/Eigen>
#include <ceres/rotation.h>

#include <opengv2/spline/BsplineReal.hpp>
#include <opengv2/bundle_adjustment/BundleAdjustmentBase.hpp>

namespace opengv2 {
    class SplineBundleAdjustment : public BundleAdjustmentBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SplineBundleAdjustment(bool fixTsb, bool robust, bool RtConstrain,
                               bool planarConstrain, double RtConstrainWeight = 3);

        void
        run(const std::map<double, Bodyframe::Ptr> &keyframes,
            const std::map<int, LandmarkBase::Ptr> &landmarks) override;

        struct ReprojectionError {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            ReprojectionError(const Eigen::Ref<const Eigen::Vector3d> &bearingVector,
                              std::shared_ptr<std::vector<std::vector<double>>> basis,
                              bool isMono)
                    : isMono_(isMono), bearingVector_(bearingVector), basis_(std::move(basis)) {}

            template<typename T>
            bool operator()(const T *const cp0, const T *const cp1, const T *const cp2, const T *const cp3,
                            const T *const lm, const T *const Tcb, T *residuals) const {
                // Map to Eigen type
                Eigen::Map<Eigen::Matrix<T, 7, 1> const> const cp_0(cp0);
                Eigen::Map<Eigen::Matrix<T, 7, 1> const> const cp_1(cp1);
                Eigen::Map<Eigen::Matrix<T, 7, 1> const> const cp_2(cp2);
                Eigen::Map<Eigen::Matrix<T, 7, 1> const> const cp_3(cp3);
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const lm_e(lm);
                Eigen::Quaternion<T> Qcb(Tcb);
                Qcb.normalize();
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const tcb(Tcb + 4);
                Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);

                Eigen::Matrix<T, 7, 1> Qbw_twb = (*basis_)[0][0] * cp_0 + (*basis_)[0][1] * cp_1 +
                                                 (*basis_)[0][2] * cp_2 + (*basis_)[0][3] * cp_3;
                Eigen::Map<Eigen::Quaternion<T>> Qbw(Qbw_twb.data());
                Qbw.normalize();
                Eigen::Map<Eigen::Matrix<T, 3, 1>> twb(Qbw_twb.data() + 4);

                Eigen::Matrix<T, 3, 1> Xb = Qbw * (lm_e - twb);
                Eigen::Matrix<T, 3, 1> Xc = Qcb * Xb + tcb;
                if (isMono_)
                    Xc.normalize();

                res = bearingVector_ - Xc;
                return true;
            }

            static ceres::CostFunction *
            Create(const Eigen::Ref<const Eigen::Vector3d> &bearingVector,
                   const std::shared_ptr<std::vector<std::vector<double>>> &basis,
                   bool isMono) {
                return (new ceres::AutoDiffCostFunction<ReprojectionError, 3, 7, 7, 7, 7, 3, 7>(
                        new ReprojectionError(bearingVector, basis, isMono)));
            }

            bool isMono_;
            Eigen::Vector3d bearingVector_;

            // for spline evaluation
            std::shared_ptr<std::vector<std::vector<double>>> basis_;
        };

        // The angle between 2nd row of Rbw and twb's 1st derivative should be constant
        // the constrain only work when straight motion, or translation difference in extrinsic parameter is very small
        struct ConstrainTcbConstant {
            explicit ConstrainTcbConstant(std::shared_ptr<std::vector<std::vector<double>>> basis)
                    : basis_(std::move(basis)) {}

            template<typename T>
            bool operator()(const T *const cp0, const T *const cp1, const T *const cp2,
                            const T *const cp3, T *residuals) const {
                // Map to Eigen type
                Eigen::Map<Eigen::Matrix<T, 7, 1> const> const cp_0(cp0);
                Eigen::Map<Eigen::Matrix<T, 7, 1> const> const cp_1(cp1);
                Eigen::Map<Eigen::Matrix<T, 7, 1> const> const cp_2(cp2);
                Eigen::Map<Eigen::Matrix<T, 7, 1> const> const cp_3(cp3);
                Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);

                Eigen::Matrix<T, 4, 1> Qbw_v = (*basis_)[0][0] * cp_0.head(4) + (*basis_)[0][1] * cp_1.head(4) +
                                               (*basis_)[0][2] * cp_2.head(4) + (*basis_)[0][3] * cp_3.head(4);
                Eigen::Map<Eigen::Quaternion<T>> Qbw(Qbw_v.data());
                Qbw.normalize();

                Eigen::Matrix<T, 3, 1> twb1d = (*basis_)[1][0] * cp_0.tail(3) + (*basis_)[1][1] * cp_1.tail(3) +
                                               (*basis_)[1][2] * cp_2.tail(3) + (*basis_)[1][3] * cp_3.tail(3);
                twb1d.normalize();

                // 2nd row of Rbw
                const T &Qx = Qbw.x();
                const T &Qy = Qbw.y();
                const T &Qz = Qbw.z();
                const T &Qw = Qbw.w();
                Eigen::Matrix<T, 3, 1> y(2. * (Qx * Qy + Qw * Qz), 1. - 2. * (Qx * Qx + Qz * Qz),
                                         2. * (Qy * Qz - Qw * Qx));
                y.normalize();

                res = twb1d - y;
                return true;
            }

            static ceres::CostFunction *
            Create(const std::shared_ptr<std::vector<std::vector<double>>> &basis) {
                return (new ceres::AutoDiffCostFunction<ConstrainTcbConstant, 3, 7, 7, 7, 7>(
                        new ConstrainTcbConstant(basis)));
            }

            // for spline evaluation
            std::shared_ptr<std::vector<std::vector<double>>> basis_;
        };

    protected:
        void optimize(ceres::Problem &problem,
                      const std::map<double, Bodyframe::Ptr> &keyframes,
                      const std::map<int, LandmarkBase::Ptr> &landmarks) override;

        bool fixTsb_;
        bool robust_;
        bool RtConstrain_;
        bool planarConstrain_;

        double RtConstrainWeight_;

        BsplineReal<7> Tspline_; // ceres type unit quaternion [w,x,y,z] Qbw, twb, by cubic spline
    };
}

#endif //OPENGV2_SPLINEBUNDLEADJUSTMENT_HPP
