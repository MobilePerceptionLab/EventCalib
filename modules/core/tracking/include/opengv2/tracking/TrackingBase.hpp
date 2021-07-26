//
// Created by huangkun on 2020/2/12.
//

#ifndef OPENGV2_TRACKINGBASE_HPP
#define OPENGV2_TRACKINGBASE_HPP

#include <memory>

#include <opengv2/map/MapBase.hpp>
#include <opengv2/match/MatchBase.hpp>

namespace opengv2 {
    class SystemBase;

    enum TrackingState {
        NOT_INITIALIZED,
        OK,
        LOST
    };

    // Tracking live in the main thread, called by system
    class TrackingBase {
    public:
        typedef std::shared_ptr<TrackingBase> Ptr;

        explicit TrackingBase(MapBase::Ptr map);

        virtual bool process(Bodyframe::Ptr bodyframe);

        void setSystem(SystemBase *system);

        std::shared_mutex stateMutex;
        TrackingState state;
        std::vector<FeatureIdentifier> lastMatchedFeatures;
        std::vector<MatchBase::Ptr> laseMatches; // for debugging
        std::mutex matchMutex;

        static Eigen::Vector3d
        triangulation(const Eigen::Quaterniond &Q12, const Eigen::Ref<const Eigen::Vector3d> &t12,
                      const Eigen::Ref<const Eigen::Vector3d> &bv1,
                      const Eigen::Ref<const Eigen::Vector3d> &bv2);

        virtual void createLandmark(Bodyframe::Ptr bf1, Bodyframe::Ptr bf2, std::vector<MatchBase::Ptr> &matches,
                                    bool enableCheck);

    protected:
        // Main tracking function. It is independent of the input sensor.
        virtual bool track(Bodyframe::Ptr bodyframe) = 0;

        // do not hold stateMutex inside initialization().
        virtual bool initialization(Bodyframe::Ptr bodyframe) = 0;

        MapBase::Ptr map_;

        SystemBase *system_;

        int landmarkCounter_;
        std::vector<int> activeLandmarks_;// for landmark removing
    };
}

#endif //OPENGV2_TRACKINGBASE_HPP
