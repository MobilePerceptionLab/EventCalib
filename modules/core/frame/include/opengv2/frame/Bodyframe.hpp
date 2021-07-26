//
// Created by huangkun on 2019/12/30.
//

#ifndef OPENGV2_BODYFRAME_HPP
#define OPENGV2_BODYFRAME_HPP

#include <shared_mutex>
#include <mutex>

#include <Eigen/Eigen>
#include <Eigen/StdVector>
//#include <cereal/types/vector.hpp>
//#include <cereal/types/memory.hpp>

#include <opengv2/frame/FrameBase.hpp>
#include <opengv2/utility/utility.hpp>

namespace opengv2 {
    class Bodyframe {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        typedef std::shared_ptr<Bodyframe> Ptr;

        inline static std::vector<Eigen::Matrix<double, 7, 1>> optTsb;
        std::vector<double> optT; // Qbw.coeffs(), twb: tx ty tz
        bool isKeyFrame; // not thread safe

        // maintained by MapOptimizer thread
        bool fixedInBA;

        Bodyframe(std::vector<FrameBase::Ptr> &frames, double timeStamp,
                  const Eigen::Ref<const Eigen::Vector3d> &twb = Eigen::Vector3d::Zero(),
                  const Eigen::Quaterniond &unitQwb = Eigen::Quaterniond::Identity()) :
                isKeyFrame(true), fixedInBA(false), timeStamp_(timeStamp),
                frames_(std::move(frames)), twb_(twb), unitQwb_(unitQwb) {
            optT.reserve(7);
        }

        /**
         * @note Return const reference of a non-const object not suitable for shared lock.
         */
        inline Eigen::Vector3d twb() const noexcept {
            std::shared_lock lock(poseMutex_);
            return twb_;
        }

        inline Eigen::Quaterniond unitQwb() const noexcept {
            std::shared_lock lock(poseMutex_);
            return unitQwb_;
        }

        inline Eigen::Matrix4d Twb() const noexcept {
            std::shared_lock lock(poseMutex_);
            Eigen::Matrix4d Twb = Eigen::Matrix4d::Identity();
            Twb.block<3, 3>(0, 0) = unitQwb_.toRotationMatrix();
            Twb.block<3, 1>(0, 3) = twb_;
            return Twb;
        }

        inline void setPose(const Eigen::Ref<const Eigen::Vector3d> &twb, const Eigen::Quaterniond &Qwb) {
            std::scoped_lock lock(poseMutex_);
            twb_ = twb;
            unitQwb_ = Qwb;
            unitQwb_.normalize();
        }

        inline void poseMutexLockShared() const {
            poseMutex_.lock_shared();
        }

        inline void poseMutexUnlockShared() const {
            poseMutex_.unlock_shared();
        }

        inline static Eigen::Vector3d tsb(size_t index) {
            std::shared_lock lock(mutexTsb_);
            return tsb_[index];
        }

        inline static Eigen::Quaterniond unitQsb(size_t index) {
            std::shared_lock lock(mutexTsb_);
            return unitQsb_[index];
        }

        inline static void
        setTsb(size_t index, const Eigen::Ref<const Eigen::Vector3d> &tsb, const Eigen::Quaterniond &Qsb) {
            std::scoped_lock lock(mutexTsb_);
            tsb_[index] = tsb;
            unitQsb_[index] = Qsb;
            unitQsb_[index].normalize();
        }

        inline static void
        setTsb(std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> &unitQsb,
               std::vector<Eigen::Vector3d> &tsb) {
            std::scoped_lock lock(mutexTsb_);
            tsb_ = std::move(tsb);
            unitQsb_ = std::move(unitQsb);
        }

        inline FrameBase::Ptr frame(size_t index) const noexcept {
            return frames_[index];
        }

        inline static size_t size() noexcept {
            return unitQsb_.size();
        }

        inline double timeStamp() const noexcept {
            return timeStamp_;
        }

        /*// This method lets cereal know which data members to serialize
        template<class Archive>
        void serialize(Archive &archive) {
            // serialize things by passing them to the archive
            archive(timeStamp_, frames_, twb_, unitQwb_);
        }*/

    protected:
        double timeStamp_;
        std::vector<FrameBase::Ptr> frames_;

        Eigen::Vector3d twb_;
        Eigen::Quaterniond unitQwb_;
        mutable std::shared_mutex poseMutex_;

        // since c++17, inline static member don't need outside definition
        inline static std::vector<Eigen::Vector3d> tsb_;
        inline static std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> unitQsb_;
        inline static std::shared_mutex mutexTsb_;
    };
}

#endif //OPENGV2_BODYFRAME_HPP
