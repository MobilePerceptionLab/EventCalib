//
// Created by huangkun on 2020/2/28.
//

#ifndef OPENGV2_SPLINEBUNDLEADJUSTMENTV2_HPP
#define OPENGV2_SPLINEBUNDLEADJUSTMENTV2_HPP

#include <Eigen/Eigen>
#include <ceres/rotation.h>

#include <opengv2/bundle_adjustment/BundleAdjustmentBase.hpp>
#include <opengv2/spline/BsplineReal.hpp>

namespace opengv2 {
    class SplineBundleAdjustmentV2 : public BundleAdjustmentBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SplineBundleAdjustmentV2(bool fixTsb, bool robust);

        void
        run(const std::map<double, Bodyframe::Ptr> &keyframes,
            const std::map<int, LandmarkBase::Ptr> &landmarks) override;

        template<class T>
        static inline void
        derToRotMatrix(const Eigen::Matrix<T, 3, 1> &unitY, const T &theta, Eigen::Matrix<T, 3, 3> &Rbw) {
            Rbw.row(1) = unitY.transpose();
            Eigen::Matrix<T, 3, 1> n = unitY.cross(Eigen::Vector3d(0, 0, 1));
            n.normalize();
            Eigen::Matrix<T, 3, 1> z = n.cross(unitY);

            Rbw.row(2) = (ceres::cos(theta) * z + ceres::sin(theta) * n).transpose();
            Rbw.row(2) /= Rbw.row(2).norm();
            Rbw.row(0) = Rbw.row(1).cross(Rbw.row(2)); // X = Y x Z
        }

        static inline void rotMatrixToDer(const Eigen::Matrix3d &Rbw, Eigen::Vector3d &unitY, double &theta) {
            unitY = Rbw.row(1).transpose();

            Eigen::Vector3d n = unitY.cross(Eigen::Vector3d(0, 0, 1));
            n /= n.norm();
            Eigen::Vector3d z = n.cross(unitY);
            z /= z.norm();

            double tmp = Rbw.row(2) * z;
            theta = std::acos(tmp > 1.0 ? 1.0 : tmp);// [0, pi]

            if (Rbw.row(2) * n < 0)
                theta = -theta;
        }


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
                Eigen::Map<Eigen::Matrix<T, 4, 1> const> const cp_0(cp0);
                Eigen::Map<Eigen::Matrix<T, 4, 1> const> const cp_1(cp1);
                Eigen::Map<Eigen::Matrix<T, 4, 1> const> const cp_2(cp2);
                Eigen::Map<Eigen::Matrix<T, 4, 1> const> const cp_3(cp3);
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const lm_e(lm);
                Eigen::Quaternion<T> Qcb(Tcb);
                Qcb.normalize();
                Eigen::Map<Eigen::Matrix<T, 3, 1> const> const tcb(Tcb + 4);
                Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);

                // spline evaluation
                Eigen::Matrix<T, 4, 1> twb_theta = basis_->at(0)[0] * cp_0 + basis_->at(0)[1] * cp_1 +
                                                   basis_->at(0)[2] * cp_2 + basis_->at(0)[3] * cp_3;
                Eigen::Matrix<T, 3, 1> y = basis_->at(1)[0] * cp_0.head(3) + basis_->at(1)[1] * cp_1.head(3) +
                                           basis_->at(1)[2] * cp_2.head(3) + basis_->at(1)[3] * cp_3.head(3);
                y.normalize();

                Eigen::Matrix<T, 3, 3> Rbw;
                derToRotMatrix(y, twb_theta[3], Rbw);

                Eigen::Matrix<T, 3, 1> Xb = Rbw * (lm_e - twb_theta.head(3));
                Eigen::Matrix<T, 3, 1> Xc = Qcb * Xb + tcb;
                if (isMono_)
                    Xc.normalize();

                res = bearingVector_ - Xc;
                return true;
            }

            static ceres::CostFunction *
            Create(const Eigen::Ref<const Eigen::Vector3d> &bearingVector,
                   const std::shared_ptr<std::vector<std::vector<double>>> &basis, bool isMono) {
                return (new ceres::AutoDiffCostFunction<ReprojectionError, 3, 4, 4, 4, 4, 3, 7>(
                        new ReprojectionError(bearingVector, basis, isMono)));
            }

            bool isMono_;
            Eigen::Vector3d bearingVector_;

            // for spline evaluation
            std::shared_ptr<std::vector<std::vector<double>>> basis_;
        };

    protected:
        void optimize(ceres::Problem &problem,
                      const std::map<double, Bodyframe::Ptr> &keyframes,
                      const std::map<int, LandmarkBase::Ptr> &landmarks) override;

        bool fixTsb_;
        bool robust_;

        BsplineReal<4> Tspline_; // twb[3] + angle, by cubic spline
    };
}

#endif //OPENGV2_SPLINEBUNDLEADJUSTMENTV2_HPP
