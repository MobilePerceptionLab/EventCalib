//
// Created by huangkun on 2020/10/11.
//

#ifndef OPENGV2_EVENTCALIBSPLINE_HPP
#define OPENGV2_EVENTCALIBSPLINE_HPP

#include <ceres/rotation.h>

#include <opengv2/spline/BsplineReal.hpp>
#include <opengv2/spline/BsplineSO3.hpp>
#include <opengv2/map/MapBase.hpp>
#include <opengv2/event/EventContainer.hpp>
#include <opengv2/utility/utility.hpp>

namespace opengv2 {
    class EventCalibSpline {
    public:
        EventCalibSpline(MapBase::Ptr map, EventContainer::Ptr eventContainer, bool useSO3, bool reduceMap,
                         double motionTimeStep, double circleRadius);

        // The intrinsics need to get combined into a single parameter block;
        // use these enums to index instead of numeric constants.
        enum {
            OFFSET_FOCAL_LENGTH_X,
            OFFSET_FOCAL_LENGTH_Y,
            OFFSET_PRINCIPAL_POINT_X,
            OFFSET_PRINCIPAL_POINT_Y,
            OFFSET_K1,
            OFFSET_K2,
            OFFSET_K3,
            OFFSET_K4,
            OFFSET_K5,
        };

        template<typename T>
        static inline void unDistort(const T &focal_length_x,
                                     const T &focal_length_y,
                                     const T &principal_point_x,
                                     const T &principal_point_y,
                                     const T &k1,
                                     const T &k2,
                                     const T &k3,
                                     const T &k4,
                                     const T &k5,
                                     const Eigen::Vector2d &distorted_p,
                                     Sophus::Vector3<T> &Xc) {
            Xc[0] = (distorted_p[0] - principal_point_x) / focal_length_x;
            Xc[1] = (distorted_p[1] - principal_point_y) / focal_length_y;
            Xc[2] = T(1);

            T xx = Xc[0] * Xc[0];
            T yy = Xc[1] * Xc[1];
            T r2 = xx + yy;
            T r4 = r2 * r2;
            T r6 = r4 * r2;
            T r8 = r6 * r2;
            T r10 = r8 * r2;
            T r_coeff = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10;

            Xc[0] *= r_coeff;
            Xc[1] *= r_coeff;
        }

        struct CalibReprojectionError_SO3 {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            CalibReprojectionError_SO3(const Eigen::Vector2d &obs,
                                       const Eigen::Vector3d &lm, const double &radius,
                                       const Eigen::Quaterniond &Qbs, const Eigen::Vector3d &tbs,
                                       std::shared_ptr<std::vector<std::vector<double>>> rBasis,
                                       std::shared_ptr<std::vector<std::vector<double>>> tBasis)
                    : obs_(obs), lm_(lm), Qbs_(Qbs), tbs_(tbs), radius_(radius), rBasis_(rBasis), tBasis_(tBasis) {}

            template<typename T>
            bool operator()(const T *const intrinsics,
                            const T *const r_cp_0, const T *const r_cp_1, const T *const r_cp_2, const T *const r_cp_3,
                            const T *const t_cp_0, const T *const t_cp_1, const T *const t_cp_2, const T *const t_cp_3,
                            T *residuals) const {
                // Unpack the intrinsics.
                const T &focal_length_x = intrinsics[OFFSET_FOCAL_LENGTH_X];
                const T &focal_length_y = intrinsics[OFFSET_FOCAL_LENGTH_Y];
                const T &principal_point_x = intrinsics[OFFSET_PRINCIPAL_POINT_X];
                const T &principal_point_y = intrinsics[OFFSET_PRINCIPAL_POINT_Y];
                const T &k1 = intrinsics[OFFSET_K1];
                const T &k2 = intrinsics[OFFSET_K2];
                const T &k3 = intrinsics[OFFSET_K3];
                const T &k4 = intrinsics[OFFSET_K4];
                const T &k5 = intrinsics[OFFSET_K5];

                // Map to Eigen type
                Eigen::Map<Sophus::SO3<T> const> const r_cp0(r_cp_0);
                Eigen::Map<Sophus::SO3<T> const> const r_cp1(r_cp_1);
                Eigen::Map<Sophus::SO3<T> const> const r_cp2(r_cp_2);
                Eigen::Map<Sophus::SO3<T> const> const r_cp3(r_cp_3);
                Eigen::Map<Sophus::Vector3<T> const> const t_cp0(t_cp_0);
                Eigen::Map<Sophus::Vector3<T> const> const t_cp1(t_cp_1);
                Eigen::Map<Sophus::Vector3<T> const> const t_cp2(t_cp_2);
                Eigen::Map<Sophus::Vector3<T> const> const t_cp3(t_cp_3);

                // spline evaluation
                Sophus::SO3<T> Qwb = r_cp0;
                Qwb *= Sophus::SO3<T>::exp(rBasis_->at(0)[0] * ((r_cp0.inverse() * r_cp1).log())); // j=1
                Qwb *= Sophus::SO3<T>::exp(rBasis_->at(0)[1] * ((r_cp1.inverse() * r_cp2).log())); // j=2
                Qwb *= Sophus::SO3<T>::exp(rBasis_->at(0)[2] * ((r_cp2.inverse() * r_cp3).log())); // j=3
                Sophus::Vector3<T> twb = tBasis_->at(0)[0] * t_cp0 + tBasis_->at(0)[1] * t_cp1 +
                                         tBasis_->at(0)[2] * t_cp2 + tBasis_->at(0)[3] * t_cp3;

                // undistortion
                Sophus::Vector3<T> Xc;
                unDistort(focal_length_x, focal_length_y,
                          principal_point_x, principal_point_y,
                          k1, k2, k3, k4, k5,
                          obs_, Xc);

                Eigen::Quaternion<T> Qws = Qwb.unit_quaternion() /* * Qbs_.cast<T>() */; // since Tsb is identity.
                Sophus::Vector3<T> tws = /* Qwb.unit_quaternion() * tbs_.cast<T>() + */ twb;

                T tx = 2. * Qws.x();
                T ty = 2. * Qws.y();
                T tz = 2. * Qws.z();
                T twx = tx * Qws.w();
                T twy = ty * Qws.w();
                T txx = tx * Qws.x();
                T txz = tz * Qws.x();
                T tyy = ty * Qws.y();
                T tyz = tz * Qws.y();
                Sophus::Vector3<T> Rws_r2(txz - twy, tyz + twx, 1.0 - (txx + tyy));
                T depth = -tws[2] / (Rws_r2.dot(Xc));
                Xc *= depth;

                Sophus::Vector3<T> Xw = Qws * Xc + tws;
                residuals[0] = (Xw - lm_).norm() - radius_;
                return true;
            }

            static ceres::CostFunction *
            Create(const Eigen::Vector2d &obs,
                   const Eigen::Vector3d &lm, const double &radius,
                   const Eigen::Quaterniond &Qbs, const Eigen::Vector3d &tbs,
                   std::shared_ptr<std::vector<std::vector<double>>> rBasis,
                   std::shared_ptr<std::vector<std::vector<double>>> tBasis) {
                return (new ceres::AutoDiffCostFunction<CalibReprojectionError_SO3, 1, 9,
                        Sophus::SO3d::num_parameters, Sophus::SO3d::num_parameters, Sophus::SO3d::num_parameters, Sophus::SO3d::num_parameters,
                        3, 3, 3, 3>(new CalibReprojectionError_SO3(obs, lm, radius, Qbs, tbs, rBasis, tBasis)));
            }

            const Eigen::Vector2d &obs_;
            const Eigen::Vector3d &lm_;
            const Eigen::Quaterniond &Qbs_;
            const Eigen::Vector3d &tbs_;

            const double &radius_;

            std::shared_ptr<std::vector<std::vector<double>>> rBasis_, tBasis_; // rBasis_: \beta_{k, k-p+j}, j \in [1, p].
        };

        struct CalibReprojectionError {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            CalibReprojectionError(const Eigen::Vector2d &obs,
                                   const Eigen::Vector3d &lm, const double &radius,
                                   const Eigen::Quaterniond &Qbs, const Eigen::Vector3d &tbs,
                                   std::shared_ptr<std::vector<std::vector<double>>> rBasis,
                                   std::shared_ptr<std::vector<std::vector<double>>> tBasis)
                    : obs_(obs), lm_(lm), Qbs_(Qbs), tbs_(tbs), radius_(radius), rBasis_(rBasis), tBasis_(tBasis) {}

            template<typename T>
            bool operator()(const T *const intrinsics,
                            const T *const r_cp_0, const T *const r_cp_1, const T *const r_cp_2, const T *const r_cp_3,
                            const T *const t_cp_0, const T *const t_cp_1, const T *const t_cp_2, const T *const t_cp_3,
                            T *residuals) const {
                // Unpack the intrinsics.
                const T &focal_length_x = intrinsics[OFFSET_FOCAL_LENGTH_X];
                const T &focal_length_y = intrinsics[OFFSET_FOCAL_LENGTH_Y];
                const T &principal_point_x = intrinsics[OFFSET_PRINCIPAL_POINT_X];
                const T &principal_point_y = intrinsics[OFFSET_PRINCIPAL_POINT_Y];
                const T &k1 = intrinsics[OFFSET_K1];
                const T &k2 = intrinsics[OFFSET_K2];
                const T &k3 = intrinsics[OFFSET_K3];
                const T &k4 = intrinsics[OFFSET_K4];
                const T &k5 = intrinsics[OFFSET_K5];

                // Map to Eigen type
                Eigen::Map<Sophus::Vector4<T> const> const r_cp0(r_cp_0);
                Eigen::Map<Sophus::Vector4<T> const> const r_cp1(r_cp_1);
                Eigen::Map<Sophus::Vector4<T> const> const r_cp2(r_cp_2);
                Eigen::Map<Sophus::Vector4<T> const> const r_cp3(r_cp_3);
                Eigen::Map<Sophus::Vector3<T> const> const t_cp0(t_cp_0);
                Eigen::Map<Sophus::Vector3<T> const> const t_cp1(t_cp_1);
                Eigen::Map<Sophus::Vector3<T> const> const t_cp2(t_cp_2);
                Eigen::Map<Sophus::Vector3<T> const> const t_cp3(t_cp_3);

                // spline evaluation
                Sophus::Vector4<T> Qwb_v = rBasis_->at(0)[0] * r_cp0 + rBasis_->at(0)[1] * r_cp1 +
                                           rBasis_->at(0)[2] * r_cp2 + rBasis_->at(0)[3] * r_cp3;
                Qwb_v.normalize();
                Eigen::Map<Eigen::Quaternion<T>> Qwb(Qwb_v.data()); // x y z w

                Sophus::Vector3<T> twb = tBasis_->at(0)[0] * t_cp0 + tBasis_->at(0)[1] * t_cp1 +
                                         tBasis_->at(0)[2] * t_cp2 + tBasis_->at(0)[3] * t_cp3;

                // undistortion
                Sophus::Vector3<T> Xc;
                unDistort(focal_length_x, focal_length_y,
                          principal_point_x, principal_point_y,
                          k1, k2, k3, k4, k5,
                          obs_, Xc);

                Eigen::Quaternion<T> Qws = Qwb /* * Qbs_.cast<T>() */; // since Tsb is identity.
                Sophus::Vector3<T> tws = /* Qwb * tbs_.cast<T>() + */ twb;

                T tx = 2. * Qws.x();
                T ty = 2. * Qws.y();
                T tz = 2. * Qws.z();
                T twx = tx * Qws.w();
                T twy = ty * Qws.w();
                T txx = tx * Qws.x();
                T txz = tz * Qws.x();
                T tyy = ty * Qws.y();
                T tyz = tz * Qws.y();
                Sophus::Vector3<T> Rws_r2(txz - twy, tyz + twx, 1.0 - (txx + tyy));
                T depth = -tws[2] / (Rws_r2.dot(Xc));
                Xc *= depth;

                Sophus::Vector3<T> Xw = Qws * Xc + tws;
                residuals[0] = (Xw - lm_).norm() - radius_;
                return true;
            }

            static ceres::CostFunction *
            Create(const Eigen::Vector2d &obs,
                   const Eigen::Vector3d &lm, const double &radius,
                   const Eigen::Quaterniond &Qbs, const Eigen::Vector3d &tbs,
                   std::shared_ptr<std::vector<std::vector<double>>> rBasis,
                   std::shared_ptr<std::vector<std::vector<double>>> tBasis) {
                return (new ceres::AutoDiffCostFunction<CalibReprojectionError, 1, 9,
                        4, 4, 4, 4,
                        3, 3, 3, 3>(new CalibReprojectionError(obs, lm, radius, Qbs, tbs, rBasis, tBasis)));
            }

            const Eigen::Vector2d &obs_;
            const Eigen::Vector3d &lm_;
            const Eigen::Quaterniond &Qbs_;
            const Eigen::Vector3d &tbs_;

            const double &radius_;

            std::shared_ptr<std::vector<std::vector<double>>> rBasis_, tBasis_;
        };

        /**
         * @brief clean redundant segments:
         *  only 1st time, use spline gap threshold1 to segmenting.
         * 1st. clean segment: sample num less than degree+1.
         * 2nd. duplicate segment: (similar t[kd-tree]; similar R[kd-tree];
         *  a. segment contain constant part: delete such pieces.
         *  b. a segment has similar part itself: keep one.
         *  c. segments have similar part: keep one, delete others.
         *  [all part duration > threshold2, otherwise not delete]).
         * 3rd. clean segment: sample num less than degree+1.
         * 4th. use these segments to initialize splines.
         * @todo Seems not work, to be improved.
         */
        void reduceMap();

        bool optimize();

        void updateMap();

        /**
         * @brief store relative variables for calculating back-projection error
         */
        struct RelationContainer {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            RelationContainer(const Eigen::Vector2d &obs, const Eigen::Vector3d &lm,
                              std::shared_ptr<std::vector<std::vector<double>>> rBasis,
                              std::shared_ptr<std::vector<std::vector<double>>> tBasis,
                              size_t rSpanIdx, size_t tSpanIdx, size_t splineIdx)
                    : obs(obs), lm(lm), rBasis(rBasis), tBasis(tBasis),
                      rSpanIdx(rSpanIdx), tSpanIdx(tSpanIdx), splineIdx(splineIdx) {}

            const Eigen::Vector2d obs;
            const Eigen::Vector3d lm;
            std::shared_ptr<std::vector<std::vector<double>>> rBasis;
            std::shared_ptr<std::vector<std::vector<double>>> tBasis;
            size_t rSpanIdx;
            size_t tSpanIdx;
            size_t splineIdx;
        };

        inline int time2splineIdx(double timestamp) const noexcept {
            auto idxItr = std::find_if(time2splineIdx_.begin(), time2splineIdx_.end(),
                                       [&timestamp](const std::pair<double, double> &p) {
                                           return (p.first <= timestamp) && (p.second >= timestamp);
                                       });
            if (idxItr == time2splineIdx_.end()) {
                return -1;
            } else {
                return idxItr - time2splineIdx_.begin();
            }
        }

        inline bool evaluate(double timestamp, Eigen::Quaterniond &unitQwb, Eigen::Vector3d &twb) const noexcept {
            int idx = time2splineIdx(timestamp);
            if (idx < 0)
                return false;

            vectorofEigenMatrix<Eigen::Vector3d> Ders;
            twbSplines_[idx].evaluate(timestamp, 0, Ders);
            twb = Ders[0];

            if (useSO3_) {
                Sophus::SO3d Rwb;
                std::vector<Eigen::Vector3d> unused;
                QwbSO3Splines_[idx].evaluate(timestamp, 0, Rwb, unused);
                unitQwb = Rwb.unit_quaternion();
            } else {
                vectorofEigenMatrix<Eigen::Vector4d> ders;
                QwbSplines_[idx].evaluate(timestamp, 0, ders);
                ders[0].normalize();
                Eigen::Map<Eigen::Quaterniond> q(ders[0].data());
                unitQwb = q;
            }

            return true;
        }

        inline const std::vector<std::vector<double>> &segments() const {
            return sampleIdSets_;
        }

    protected:
        MapBase::Ptr map_;
        EventContainer::Ptr eventContainer_;

        std::vector<std::pair<double, double>> time2splineIdx_;

        const int degree_ = 3;
        // TODO: improve ceres automatic-diff function parameter num limitation(use SE3?) for higher degree
        std::vector<BsplineReal<3>> twbSplines_; // degree = 3

        bool useSO3_;
        std::vector<BsplineSO3> QwbSO3Splines_; // degree = 3
        std::vector<BsplineReal<4>> QwbSplines_; // degree = 3

        Eigen::Matrix<double, 9, 1> intrinsics_;

        double motionTimeStep_;
        double circleRadius_;

        std::vector<RelationContainer> relationContainer_; // used for calculating back-projection error

        std::vector<std::vector<double>> sampleIdSets_;
        bool reduceMap_; // reduce map by similarity check
    };
}

#endif //OPENGV2_EVENTCALIBSPLINE_HPP
