//
// Created by huangkun on 2020/1/9.
//

#include <iostream>

#include <opengv2/map/MapBase.hpp>

opengv2::MapBase::MapBase() : subMap(nullptr) {}

void opengv2::MapBase::removeFrame(double timestamp) {
    // clear links
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
    }

    MapBase *map = this;
    while (map != nullptr) {
        if (map->hasKeyframe(timestamp)) {
            map->keyframesMutex_.lock();
            map->keyframes_.erase(timestamp);
            map->keyframesMutex_.unlock();

            map = map->subMap.get();
        } else {
            break;
        }
    }
}

void opengv2::MapBase::removeLandmark(int id) {
    if (auto lm = landmark(id))
        lm->clearObservations();

    MapBase *map = this;
    while (map != nullptr) {
        if (map->hasLandmark(id)) {
            map->landmarksMutex_.lock();
            map->landmarks_.erase(id);
            map->landmarksMutex_.unlock();

            map = map->subMap.get();
        } else {
            break;
        }
    }
}

void opengv2::MapBase::switchOptData(const std::map<double, Bodyframe::Ptr> &keyframes,
                                     const std::map<int, LandmarkBase::Ptr> &landmarks) {
    for (const auto &it: keyframes) {
        auto &bf = it.second;

        Eigen::Quaterniond Qwb = bf->unitQwb();
        Eigen::Vector3d twb = bf->twb();

        if (!bf->optT.empty()) {
            // opt -> real
            Eigen::Quaterniond Qbw(bf->optT.data());
            Qbw.normalize();
            bf->setPose(Eigen::Vector3d(bf->optT.data() + 4), Qbw.conjugate());
        }

        // real -> opt
        bf->optT.clear();
        Eigen::Quaterniond Qbw = Qwb.conjugate();
        std::copy(Qbw.coeffs().data(), Qbw.coeffs().data() + 4, std::back_inserter(bf->optT));
        std::copy(twb.data(), twb.data() + 3, std::back_inserter(bf->optT));
    }
    for (const auto &it: landmarks) {
        auto lm = it.second;
        Eigen::Vector3d p = lm->position();

        if (!lm->optData.empty()) {
            // opt -> real
            lm->setPosition(Eigen::Vector3d(lm->optData.data()));
        }

        // real -> opt
        lm->optData.clear();
        std::copy(p.data(), p.data() + 3, std::back_inserter(lm->optData));
    }
    for (size_t i = 0; i < Bodyframe::size(); i++) {
        Eigen::Quaterniond Qsb = Bodyframe::unitQsb(i);
        Eigen::Vector3d tsb = Bodyframe::tsb(i);

        if (i < Bodyframe::optTsb.size()) {
            // opt -> real
            Bodyframe::setTsb(i, Bodyframe::optTsb[i].tail<3>(), Eigen::Quaterniond(Bodyframe::optTsb[i].data()));

            // real -> opt
            Bodyframe::optTsb[i].head<4>() = Qsb.coeffs();
            Bodyframe::optTsb[i].tail<3>() = tsb;
        } else {
            // real -> opt
            Eigen::Matrix<double, 7, 1> T;
            T.head<4>() = Qsb.coeffs();
            T.tail<3>() = tsb;
            Bodyframe::optTsb.push_back(T);
        }
    }
}

void opengv2::MapBase::clear() {
    MapBase *map = this;
    while (map != nullptr) {
        std::unique_lock lock1(map->keyframesMutex_);
        std::unique_lock lock2(map->landmarksMutex_);
        std::unique_lock lock3(map->landmarksToBeErasedMutex_);
        std::unique_lock lock4(map->keyFramesToBeErasedMutex_);

        map->keyframes_.clear();
        map->landmarks_.clear();
        map->landmarksToBeErased_.clear();
        map->keyFramesToBeErased_.clear();

        map = map->subMap.get();
    }
}
