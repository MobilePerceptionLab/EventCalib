//
// Created by huangkun on 2019/12/30.
//

#ifndef OPENGV2_FEATUREBASE_HPP
#define OPENGV2_FEATUREBASE_HPP

#include <memory>

#include <Eigen/Eigen>

#include <opengv2/landmark/LandmarkBase.hpp>

namespace opengv2 {
    class FeatureBase { // TODO: inherit ObservationBase
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef std::shared_ptr<FeatureBase> Ptr;

        explicit FeatureBase(const Eigen::Ref<const Eigen::VectorXd> &loc) :
                locBackup(loc), landmark_(LandmarkBase::Ptr()), loc_(loc), bearingVector_(Eigen::VectorXd()) {}

        virtual ~FeatureBase() = default;

        inline LandmarkBase::Ptr landmark() const noexcept {
            return landmark_.lock();
        }

        /**
         * @note Warning: Not thread safe.
         */
        inline void setLandmark(const LandmarkBase::Ptr &lm) noexcept {
            landmark_ = lm;
        }

        inline const Eigen::VectorXd &location() const noexcept {
            return loc_;
        }

        /**
         * @note Warning: Not thread safe.
         */
        virtual inline void setLocation(const Eigen::Ref<const Eigen::VectorXd> &loc) noexcept {
            loc_ = loc;
        }

        inline const Eigen::VectorXd &bearingVector() noexcept {
            return bearingVector_;
        }

        /**
         * @note Warning: Not thread safe.
         */
        inline void setBearingVector(const Eigen::Ref<const Eigen::VectorXd> &bv) {
            bearingVector_ = bv;
        }

        Eigen::VectorXd locBackup; // backup

    protected:
        std::weak_ptr<LandmarkBase> landmark_;
        Eigen::VectorXd loc_;
        Eigen::VectorXd bearingVector_;
    };
}

#endif //OPENGV2_FEATUREBASE_HPP
