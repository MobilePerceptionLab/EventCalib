//
// Created by huangkun on 2020/1/9.
//

#ifndef OPENGV2_MAPBASE_HPP
#define OPENGV2_MAPBASE_HPP

#include <shared_mutex>
#include <map>
#include <set>
#include <memory>

#include <opengv2/landmark/LandmarkBase.hpp>
#include <opengv2/frame/Bodyframe.hpp>
#include <opengv2/frame/CameraFrame.hpp>

namespace opengv2 {
    class MapBase {
    public:
        typedef std::shared_ptr<MapBase> Ptr;

        MapBase();

        inline bool hasKeyframe(double timestamp) const {
            std::shared_lock lock(keyframesMutex_);
            return keyframes_.find(timestamp) != keyframes_.end();
        }

        inline bool hasLandmark(int id) const {
            std::shared_lock lock(landmarksMutex_);
            return landmarks_.find(id) != landmarks_.end();
        }

        inline void addFrame(const Bodyframe::Ptr &bf) {
            std::unique_lock lock(keyframesMutex_);
            keyframes_[bf->timeStamp()] = bf;
        }

        inline void addLandmark(const LandmarkBase::Ptr &landmark) {
            std::unique_lock lock(landmarksMutex_);
            landmarks_[landmark->id()] = landmark;
        }

        void removeFrame(double timestamp);

        void removeLandmark(int id);

        inline Bodyframe::Ptr keyframe(double timestamp) const {
            std::shared_lock lock(keyframesMutex_);
            auto result = keyframes_.find(timestamp);
            return result == keyframes_.end() ? nullptr : result->second;
        }

        inline LandmarkBase::Ptr landmark(int id) const {
            std::shared_lock lock(landmarksMutex_);
            auto result = landmarks_.find(id);
            return result == landmarks_.end() ? nullptr : result->second;
        }

        inline Bodyframe::Ptr firstKeyframe() const {
            std::shared_lock lock(keyframesMutex_);
            if (keyframes_.empty())
                return nullptr;
            else
                return keyframes_.cbegin()->second;
        }

        inline Bodyframe::Ptr lastKeyframe() const {
            std::shared_lock lock(keyframesMutex_);
            if (keyframes_.empty())
                return nullptr;
            else
                return keyframes_.crbegin()->second;
        }

        inline bool empty() const {
            std::shared_lock lock(keyframesMutex_);
            return keyframes_.empty();
        }

        /**
         * @brief clear map
         */
        void clear();

        inline int frameNum() const {
            std::shared_lock lock(keyframesMutex_);
            return keyframes_.size();
        }

        inline int landmarkNum() const {
            std::shared_lock lock(landmarksMutex_);
            return landmarks_.size();
        }

        /**
         * @brief require shared mutex when using.
         */
        inline const std::map<double, Bodyframe::Ptr> &keyframes() const {
            return keyframes_;
        }

        inline std::map<double, Bodyframe::Ptr> copyKeyframes() const {
            std::shared_lock lock(keyframesMutex_);
            return keyframes_;
        }

        /**
         * @brief require shared mutex when using.
         */
        inline const std::map<int, LandmarkBase::Ptr> &landmarks() const {
            return landmarks_;
        }

        inline std::map<int, LandmarkBase::Ptr> copyLandmarks() const {
            std::shared_lock lock(landmarksMutex_);
            return landmarks_;
        }

        inline void keyframeLockShared() const {
            keyframesMutex_.lock_shared();
        }

        inline void keyframeUnlockShared() const {
            keyframesMutex_.unlock_shared();
        }

        inline void landmarkLockShared() const {
            landmarksMutex_.lock_shared();
        }

        void landmarkUnlockShared() const {
            landmarksMutex_.unlock_shared();
        }

        /**
         * @brief Erase landmarks and keyframes in RemoveList.
         */
        inline void cleanMap() {
            landmarksToBeErasedMutex_.lock();
            for (int lmId: landmarksToBeErased_) {
                removeLandmark(lmId);
            }
            landmarksToBeErased_.clear();
            landmarksToBeErasedMutex_.unlock();
            keyFramesToBeErasedMutex_.lock();
            for (double timeStamp: keyFramesToBeErased_) {
                removeFrame(timeStamp);
            }
            keyFramesToBeErased_.clear();
            keyFramesToBeErasedMutex_.unlock();
        }

        /**
         * @brief Add landmark to RemoveList. Remember to clear observations.
         */
        inline void requestRemoveLandmark(int lmId) {
            if (auto lm = landmark(lmId)) {
                lm->clearObservations();

                std::scoped_lock lock(landmarksToBeErasedMutex_);
                landmarksToBeErased_.push_back(lmId);
            }
        }

        /**
         * @brief Add keyframe to RemoveList. Remember to clear observations.
         */
        inline void requestRemoveKeyframe(double timestamp) {
            if (auto bf = keyframe(timestamp)) {
                for (int i = 0; i < bf->size(); ++i) {
                    if (auto cf = dynamic_cast<CameraFrame *>(bf->frame(i).get())) {
                        for (const auto &f: cf->features()) {
                            if (auto lm = f->landmark()) {
                                lm->eraseObservation(timestamp);
                            }
                        }
                    }
                }

                std::scoped_lock lock(keyFramesToBeErasedMutex_);
                keyFramesToBeErased_.push_back(timestamp);
            }
        }

        static void
        switchOptData(const std::map<double, Bodyframe::Ptr> &keyframes,
                      const std::map<int, LandmarkBase::Ptr> &landmarks);

        MapBase::Ptr subMap; // parent have no mutex control over child, but part elements copied to child

    protected:
        std::map<double, Bodyframe::Ptr> keyframes_;
        std::map<int, LandmarkBase::Ptr> landmarks_;
        // these mutex only control the map, for the content in lm and bf they have their own mutex
        mutable std::shared_mutex keyframesMutex_, landmarksMutex_;

        std::vector<int> landmarksToBeErased_;
        std::vector<double> keyFramesToBeErased_;
        mutable std::mutex landmarksToBeErasedMutex_, keyFramesToBeErasedMutex_;
    };
}

#endif //OPENGV2_MAPBASE_HPP
