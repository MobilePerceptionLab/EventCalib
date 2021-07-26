//
// Created by huangkun on 2020/1/2.
//

#ifndef OPENGV2_LANDMARKBASE_HPP
#define OPENGV2_LANDMARKBASE_HPP

#include <memory>
#include <shared_mutex>
#include <mutex>

#include <Eigen/Eigen>

#include <opengv2/feature/FeatureIdentifier.hpp>

namespace opengv2 {
    class LandmarkBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        typedef std::shared_ptr<LandmarkBase> Ptr;

        LandmarkBase(int id, const Eigen::Ref<const Eigen::Vector3d> &position) :
                fixedInBA(false), id_(id), position_(position) {
            optData.reserve(3);
        }

        virtual ~LandmarkBase() = default;

        inline LandmarkBase &operator=(const LandmarkBase &other) {
            std::shared_lock lock1(observationMutex);
            std::shared_lock lock2(positionMutex_);

            id_ = other.id_;
            position_ = other.position_;
            observations_ = other.observations_;
        }

        inline int id() const noexcept {
            return id_;
        }

        inline void setId(int id) noexcept {
            id_ = id;
        }

        inline Eigen::Vector3d position() const noexcept {
            std::shared_lock lock(positionMutex_);
            return position_;
        }

        inline void setPosition(const Eigen::Ref<const Eigen::Vector3d> &p) {
            std::scoped_lock lock(positionMutex_);
            position_ = p;
        }

        /**
         * @brief use along with observationMutex
         * @note only for iteration, for other cases please use API
         */
        inline std::map<double, FeatureIdentifier> &observations() noexcept {
            return observations_;
        }

        inline bool emptyObservations() const {
            std::shared_lock lock(observationMutex);
            return observations_.empty();
        }

        inline size_t observationNum() const {
            std::shared_lock lock(observationMutex);
            return observations_.size();
        }

        inline void addObservation(const FeatureIdentifier &fi) noexcept {
            std::scoped_lock lock(observationMutex);
            observations_.insert(std::make_pair(fi.timestamp(), fi));
        }

        inline void eraseObservation(double timestamp) {
            std::scoped_lock lock(observationMutex);
            observations_.erase(timestamp);
        }

        inline void clearObservations() {
            std::scoped_lock lock(observationMutex);
            observations_.clear();
        }

        inline const FeatureIdentifier &firstObservation() const {
            std::shared_lock lock(observationMutex);
            return observations_.cbegin()->second;
        }

        inline const FeatureIdentifier &lastObservation() const {
            std::shared_lock lock(observationMutex);
            return observations_.crbegin()->second;
        }

        inline bool hasObservation(double timestamp) const {
            std::shared_lock lock(observationMutex);
            return observations_.find(timestamp) != observations_.end();
        }

        std::vector<double> optData; // not thread safe

        // maintained by map_optimizer thread
        bool fixedInBA;

        mutable std::shared_mutex observationMutex;

    protected:
        int id_;
        Eigen::Vector3d position_;
        mutable std::shared_mutex positionMutex_;

        std::map<double, FeatureIdentifier> observations_;
    };
}

#endif //OPENGV2_LANDMARKBASE_HPP
