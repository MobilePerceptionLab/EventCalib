//
// Created by huangkun on 2021/6/17.
//

#include <opengv2/map_optimizer/GVMapOptimizer.hpp>
#include <opengv2/frame/CameraFrame.hpp>

opengv2::GVMapOptimizer::GVMapOptimizer(opengv2::MapBase::Ptr map, ViewerBase::Ptr viewer,
                                        BundleAdjustmentBase::Ptr bundleAdjustment,
                                        int newFrameNumLimit) :
        MapOptimizerBase(std::move(map), std::move(viewer), std::move(bundleAdjustment), newFrameNumLimit) {}

void opengv2::GVMapOptimizer::cullingMap() {
    for (const auto &it: landmarks_) {
        auto lm = it.second;
        lm->observationMutex.lock();
        for (auto itr = lm->observations().begin(); itr != lm->observations().end();) {
            const auto &obs = itr->second;
            Bodyframe::Ptr bf = map_->keyframe(obs.timestamp());
            if (bf == nullptr) {
                itr = lm->observations().erase(itr);
                continue;
            }
            auto f = obs.feature();
            if (f == nullptr) {
                itr = lm->observations().erase(itr);
                continue;
            }
            Eigen::Vector3d Xb = bf->unitQwb().conjugate() * (lm->position() - bf->twb());
            Eigen::Vector3d Xc = Bodyframe::unitQsb(obs.frameId()) * Xb + Bodyframe::tsb(obs.frameId());
            Eigen::Vector3d bv = f->bearingVector();
            Xc.normalize();
            bv.normalize();

            // TODO: add support for stereo, depth check
            if (1.0 - (bv.transpose() * Xc) > 1 - cos(atan(3 / 800.))) {
                // remove bad observation
                f->setLandmark(nullptr);
                itr = lm->observations().erase(itr);
            } else {
                itr++;
            }
        }
        lm->observationMutex.unlock();

        if (lm->observationNum() < 2) {
            map_->requestRemoveLandmark(lm->id());
        }
    }
    map_->cleanMap();
}

double opengv2::GVMapOptimizer::selectVariableForBA() {
    landmarks_.clear();
    keyframes_.clear();

    double minTimeStamp = lastFrameStamp_;
    map_->keyframeLockShared();
    for (auto itr = map_->keyframes().lower_bound(lastFrameStamp_); itr != map_->keyframes().end(); itr++) {
        auto bf = itr->second;
        keyframes_[bf->timeStamp()] = bf;
        for (int camIndex = 0; camIndex < bf->size(); camIndex++) {
            auto cf = dynamic_cast<CameraFrame *>(bf->frame(camIndex).get());
            for (const auto &feature: cf->features()) {
                if (auto lm = feature->landmark()) { // direct landmarks
                    landmarks_[lm->id()] = lm;
                    if (!lm->emptyObservations())
                        minTimeStamp = std::min(lm->firstObservation().timestamp(), minTimeStamp);
                }
            }
        }
    }

    for (auto itr = map_->keyframes().lower_bound(minTimeStamp);
         itr != map_->keyframes().lower_bound(lastFrameStamp_); itr++) {
        auto bf = itr->second;
        keyframes_[bf->timeStamp()] = bf;
        for (int camIndex = 0; camIndex < bf->size(); camIndex++) {
            auto cf = dynamic_cast<CameraFrame *>(bf->frame(camIndex).get());
            for (const auto &feature: cf->features()) {
                if (auto lm = feature->landmark()) { // fix indirect landmarks
                    landmarks_[lm->id()] = lm;
                    if (!lm->emptyObservations()) {
                        if (lm->lastObservation().timestamp() < lastFrameStamp_) {
                            lm->fixedInBA = true;
                        }
                    }
                }
            }
        }
    }
    double endStamp = map_->keyframes().rbegin()->first;
    map_->keyframeUnlockShared();
    return endStamp;
}
