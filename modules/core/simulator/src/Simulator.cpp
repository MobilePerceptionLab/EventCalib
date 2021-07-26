//
// Created by huangkun on 2020/7/25.
//

#include <opengv2/simulator/Simulator.hpp>
#include <opengv2/bundle_adjustment/SplineBundleAdjustmentV2.hpp>
#include <opengv2/frame/CameraFrame.hpp>

opengv2::Simulator::Simulator(SensorBase::Ptr sensor, bool randomSeed) : sensor_(std::move(sensor)) {
    if (randomSeed) {
        std::random_device rd;
        gen_.seed(rd());
    } else {
        gen_.seed(0);
    }
}

void opengv2::Simulator::generateTrajectory(const MapBase::Ptr &map, double scale) {
    double timeStep = 0.1;
    for (int i = 0; i < 100; i++) {
        double timeStamp = timeStep * i;
        double t = scale * i;
        Eigen::Vector3d twb(cos(t), t, 0);
        Eigen::Vector3d y(-sin(t), 1, 0);
        y.normalize();
        Eigen::Matrix3d Rbw;
        SplineBundleAdjustmentV2::derToRotMatrix(y, 0.0, Rbw);
        Eigen::Quaterniond Qbw(Rbw);
        Qbw.normalize();
        Eigen::Quaterniond Qwb = Qbw.conjugate();

        // create frame
        std::vector<FrameBase::Ptr> frames;
        frames.push_back(std::make_shared<CameraFrame>(cv::Mat(), std::dynamic_pointer_cast<CameraBase>(sensor_)));
        // create bodyframe
        Bodyframe::Ptr bf = std::make_shared<Bodyframe>(frames, timeStamp, twb, Qwb);
        map->addFrame(bf);
    }
}

void opengv2::Simulator::clearLandmarks(const opengv2::MapBase::Ptr &map) {
    map->landmarkLockShared();
    for (const auto &pair: map->landmarks()) {
        map->requestRemoveLandmark(pair.first);
    }
    map->landmarkUnlockShared();
    map->cleanMap();
}

void opengv2::Simulator::generateLandmarks(const opengv2::MapBase::Ptr &map, int localConnectivity) {
    auto camera = dynamic_cast<CameraBase *>(sensor_.get());

    // one observation for each frame to generateTrajectory landmarks
    int landmarkCounter = 0;
    std::uniform_int_distribution<> disX(50, camera->size()[0] - 50);
    std::uniform_int_distribution<> disY(50, camera->size()[1] - 50);
    std::uniform_real_distribution<> disD(6.0, 30.0);
    std::vector<FeatureBase::Ptr> features;
    map->keyframeLockShared();
    for (const auto &itr: map->keyframes()) {
        auto bf = itr.second;
        auto cf = dynamic_cast<CameraFrame *>(bf->frame(0).get());
        for (int i = 0; i < localConnectivity; i++) {
            Eigen::Vector2d p(disX(gen_), disY(gen_));
            Eigen::Vector3d Xc = dynamic_cast<CameraBase *>(bf->frame(0)->sensor().get())->invProject(p);
            Xc = Xc / Xc.norm() * disD(gen_);
            Eigen::Vector3d Xb =
                    Bodyframe::unitQsb(0).conjugate() * Xc - Bodyframe::unitQsb(0).conjugate() * Bodyframe::tsb(0);
            Eigen::Vector3d Xw = bf->unitQwb() * Xb + bf->twb();

            // feature
            auto feature = std::make_shared<FeatureBase>(p);
            features.push_back(feature);
            FeatureIdentifier fi(bf->timeStamp(), 0, features.size() - 1, feature);

            // landmark
            LandmarkBase::Ptr lm = std::make_shared<LandmarkBase>(landmarkCounter++, Xw);
            lm->addObservation(fi);
            map->addLandmark(lm);

            // link feature and lm
            feature->setLandmark(lm);
        }
        cf->setFeatures(features);
    }
    map->keyframeUnlockShared();
}

void opengv2::Simulator::generateObs(const MapBase::Ptr &map, int globalConnectivity) {
    auto camera = dynamic_cast<CameraBase *>(sensor_.get());

    // project each landmark to each frame
    map->landmarkLockShared();
    for (const auto &it0: map->landmarks()) {
        auto lm = it0.second;
        Eigen::Vector3d Xw = lm->position();

        // 1st frame observe the landmark
        map->keyframeLockShared();
        int originId = std::distance(map->keyframes().begin(),
                                     map->keyframes().find(lm->firstObservation().timestamp()));
        map->keyframeUnlockShared();

        map->keyframeLockShared();
        for (const auto &it1: map->keyframes()) {
            auto bf = it1.second;
            map->keyframeLockShared();
            int distance = std::distance(map->keyframes().begin(), map->keyframes().find(bf->timeStamp()));
            map->keyframeUnlockShared();
            if (std::abs(distance) > (originId + globalConnectivity) ||
                std::abs(distance) < (originId - globalConnectivity))
                continue;

            Eigen::Vector3d Xb = bf->unitQwb().conjugate() * Xw - bf->unitQwb().conjugate() * bf->twb();
            Eigen::Vector3d Xc = Bodyframe::unitQsb(0) * Xb + Bodyframe::tsb(0);
            Eigen::Vector2d p = dynamic_cast<CameraBase *>(bf->frame(0)->sensor().get())->project(Xc);

            if (p(0) > 0 && p(0) < camera->size()[0] && p(1) > 0 && p(1) < camera->size()[1]) {
                // feature
                auto feature = std::make_shared<FeatureBase>(p);
                auto cf = dynamic_cast<CameraFrame *>(bf->frame(0).get());
                cf->features().push_back(feature);
                FeatureIdentifier fi(bf->timeStamp(), 0, cf->features().size() - 1, feature);

                // landmark
                lm->addObservation(fi);

                // link feature and lm
                feature->setLandmark(lm);
            }
        }
        map->keyframeUnlockShared();
    }
    map->landmarkUnlockShared();
}

void opengv2::Simulator::addNoise(const opengv2::MapBase::Ptr &map, int noiseLevel, double proportion) {
    // adding noise
    std::normal_distribution<double> obsNoise(0.0, noiseLevel); // pixel unit
    std::binomial_distribution<int> select(1, proportion);
    map->keyframeLockShared();
    for (const auto &it1: map->keyframes()) {
        auto bf = it1.second;
        auto cf = dynamic_cast<CameraFrame *>(bf->frame(0).get());
        for (auto &fb: cf->features()) {
            if (fb->landmark() != nullptr) {
                fb->setLocation(fb->locBackup + (select(gen_) ? Eigen::Vector2d(obsNoise(gen_), obsNoise(gen_))
                                                              : Eigen::Vector2d::Zero()));
            }
        }
        cf->extractBearingVectors();
    }
    map->keyframeUnlockShared();
}
