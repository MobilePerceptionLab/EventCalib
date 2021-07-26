//
// Created by huangkun on 2021/6/17.
//

#include <thread>

#include <opengv2/map_optimizer/MapOptimizerBase.hpp>

opengv2::MapOptimizerBase::MapOptimizerBase(MapBase::Ptr map, ViewerBase::Ptr viewer,
                                            BundleAdjustmentBase::Ptr bundleAdjustment, int newFrameNumLimit) :
        map_(std::move(map)), viewer_(std::move(viewer)), bundleAdjustment_(std::move(bundleAdjustment)),
        newFrameNumLimit_(newFrameNumLimit), lastFrameStamp_(std::numeric_limits<double>::min()) {
    finishRequested_ = false;
    finished_ = false;
}

void opengv2::MapOptimizerBase::run() {
    while (true) {
        map_->keyframeLockShared();
        if (!map_->keyframes().empty()) {
            if (lastFrameStamp_ == std::numeric_limits<double>::min()) {
                lastFrameStamp_ = map_->keyframes().begin()->first;
            }
            auto newFrameBegin = map_->keyframes().lower_bound(lastFrameStamp_);
            auto newFrameNum = std::distance(newFrameBegin, map_->keyframes().end());
            map_->keyframeUnlockShared();
            if (newFrameNum > newFrameNumLimit_) {// TODO: use a single incrementally spline.
                lastFrameStamp_ = localBundleAdjustment();
            }
        } else {
            map_->keyframeUnlockShared();
        }

        if (finishRequested()) {
            setFinish();
            break;
        }

        wait();
    }
}

bool opengv2::MapOptimizerBase::finishRequested() {
    std::unique_lock<std::mutex> lock(mutexFinish_);
    return finishRequested_;
}

void opengv2::MapOptimizerBase::requestFinish() {
    std::unique_lock<std::mutex> lock(mutexFinish_);
    finishRequested_ = true;
}

bool opengv2::MapOptimizerBase::isFinished() {
    std::unique_lock<std::mutex> lock(mutexFinish_);
    return finished_;
}

void opengv2::MapOptimizerBase::setFinish() {
    std::unique_lock<std::mutex> lock(mutexFinish_);
    finished_ = true;
}

void opengv2::MapOptimizerBase::clearFlagAndOptData() {
    for (auto &it: keyframes_) {
        auto &bf = it.second;
        bf->fixedInBA = false;
        bf->optT.clear();
    }
    for (auto &it: landmarks_) {
        auto lm = it.second;
        lm->fixedInBA = false;
        lm->optData.clear();
    }
    Bodyframe::optTsb.clear();
    keyframes_.clear();
    landmarks_.clear();
}

double opengv2::MapOptimizerBase::localBundleAdjustment() {
    double endStamp = selectVariableForBA();

    bundleAdjustment_->run(keyframes_, landmarks_);
    map_->switchOptData(keyframes_, landmarks_);

    cullingMap();
    clearFlagAndOptData();
    return endStamp;
}
