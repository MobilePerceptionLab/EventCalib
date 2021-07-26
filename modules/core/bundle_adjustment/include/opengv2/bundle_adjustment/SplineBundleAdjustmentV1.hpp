//
// Created by huangkun on 2020/6/23.
//

#ifndef OPENGV2_SPLINEBUNDLEADJUSTMENTV1_HPP
#define OPENGV2_SPLINEBUNDLEADJUSTMENTV1_HPP

#include <opengv2/bundle_adjustment/NormalBundleAdjustment.hpp>
#include <opengv2/spline/BsplineReal.hpp>

namespace opengv2 {
    class SplineBundleAdjustmentV1 : public NormalBundleAdjustment {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SplineBundleAdjustmentV1(bool fixTsb, bool robust, bool planarConstrain, bool RtConstrain,
                                 double splineDistanceWeight = 3, double RtConstrainWeight = 3);

        void run(const std::map<double, Bodyframe::Ptr> &keyframes,
                 const std::map<int, LandmarkBase::Ptr> &landmarks) override;

        struct SplineDistance {
            explicit SplineDistance(std::shared_ptr<std::vector<std::vector<double>>> basis)
                    : _basis(std::move(basis)) {}

            template<typename T>
            bool operator()(const T *const cp0, const T *const cp1, const T *const cp2,
                            const T *const cp3, const T *const Qbw_twb, T *residuals) const {
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const cp_0(cp0);
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const cp_1(cp1);
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const cp_2(cp2);
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const cp_3(cp3);
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const twb_d(Qbw_twb + 4);
                Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);

                Eigen::Matrix<T, 3, 1> twb = (*_basis)[0][0] * cp_0 + (*_basis)[0][1] * cp_1 +
                                             (*_basis)[0][2] * cp_2 + (*_basis)[0][3] * cp_3;
                res = twb_d - twb;
                return true;
            }

            static ceres::CostFunction *
            Create(const std::shared_ptr<std::vector<std::vector<double>>> &basis) {
                return (new ceres::AutoDiffCostFunction<SplineDistance, 3, 3, 3, 3, 3, 7>(
                        new SplineDistance(basis)));
            }

            // for spline evaluation
            std::shared_ptr<std::vector<std::vector<double>>> _basis;
        };

        // The angle between 2nd row of Rbw(KF) and twb's(spline) 1st derivative should be constant
        // the constrain only work when straight motion, or translation difference in extrinsic parameter is very small
        struct RtConstraint {
            explicit RtConstraint(std::shared_ptr<std::vector<std::vector<double>>> basis)
                    : _basis(std::move(basis)) {}

            template<typename T>
            bool operator()(const T *const cp0, const T *const cp1, const T *const cp2,
                            const T *const cp3, const T *const Qbw, T *residuals) const {
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const cp_0(cp0);
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const cp_1(cp1);
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const cp_2(cp2);
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const cp_3(cp3);
                Eigen::Quaternion<T> Qbw_e(Qbw);
                Qbw_e.normalize();
                Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);

                Eigen::Matrix<T, 3, 1> twb1d = (*_basis)[1][0] * cp_0 + (*_basis)[1][1] * cp_1 +
                                               (*_basis)[1][2] * cp_2 + (*_basis)[1][3] * cp_3;
                twb1d.normalize();

                // 2nd row of Rbw
                const T &Qx = Qbw_e.x();
                const T &Qy = Qbw_e.y();
                const T &Qz = Qbw_e.z();
                const T &Qw = Qbw_e.w();
                Eigen::Matrix<T, 3, 1> y(2. * (Qx * Qy + Qw * Qz), 1. - 2. * (Qx * Qx + Qz * Qz),
                                         2. * (Qy * Qz - Qw * Qx));
                y.normalize();

                res = twb1d - y;
                return true;
            }

            static ceres::CostFunction *
            Create(const std::shared_ptr<std::vector<std::vector<double>>> &basis) {
                return (new ceres::AutoDiffCostFunction<RtConstraint, 3, 3, 3, 3, 3, 7>(
                        new RtConstraint(basis)));
            }

            // for spline evaluation
            std::shared_ptr<std::vector<std::vector<double>>> _basis;
        };

    protected:
        void optimize(ceres::Problem &problem,
                      const std::map<double, Bodyframe::Ptr> &keyframes,
                      const std::map<int, LandmarkBase::Ptr> &landmarks) override;

        bool RtConstrain_;
        double RtConstrainWeight_;
        double splineDistanceWeight_;

        BsplineReal<3> Tspline_; // twb, by cubic spline
    };
}

#endif //OPENGV2_SPLINEBUNDLEADJUSTMENTV1_HPP
