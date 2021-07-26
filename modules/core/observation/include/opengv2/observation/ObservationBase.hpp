/*
 * Copyright (c) 2021. Kun Huang.
 * This Source Code Form is subject to the terms of the Apache License, v. 2.0.
 */

//
// Created by huangkun on 2021/7/21.
//

#ifndef OPENGV2_OBSERVATIONBASE_HPP
#define OPENGV2_OBSERVATIONBASE_HPP

#include <memory>
#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>

namespace opengv2 {
    class ObservationBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef std::shared_ptr<ObservationBase> Ptr;

        ObservationBase(const Eigen::Ref<const Eigen::MatrixXd> &measurement,
                        const Eigen::Ref<const Eigen::MatrixXd> &invCovariance) :
                measurement_(measurement), invSqrtCovariance_(invCovariance.sqrt()) {}

        virtual ~ObservationBase() = default;

        inline const Eigen::MatrixXd &measurement() const noexcept {
            return measurement_;
        }

        /**
         * @return The inverse square root of the covariance matrix
         * @note should be multiplied inside the cost function
         * The covariance matrix is a positive semi-definite matrix.
         */
        inline const Eigen::MatrixXd &invSqrtCovariance() const noexcept {
            return invSqrtCovariance_;
        }

        /**
         * @note Warning: Not thread safe. Just for Debug and simulation.
         */
        virtual inline void setMeasurement(const Eigen::Ref<const Eigen::MatrixXd> &measurement) noexcept {
            measurement_ = measurement;
        }

    protected:
        Eigen::MatrixXd measurement_;
        Eigen::MatrixXd invSqrtCovariance_;
    };
}

#endif //OPENGV2_OBSERVATIONBASE_HPP
