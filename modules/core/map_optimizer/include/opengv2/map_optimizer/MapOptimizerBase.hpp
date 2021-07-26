//
// Created by huangkun on 2021/6/16.
//

#ifndef OPENGV2_MAPOPTIMIZERBASE_HPP
#define OPENGV2_MAPOPTIMIZERBASE_HPP

#include <opengv2/map/MapBase.hpp>
#include <opengv2/bundle_adjustment/BundleAdjustmentBase.hpp>
#include <opengv2/viewer/ViewerBase.hpp>

namespace opengv2 {
    class MapOptimizerBase {
    public:
        typedef std::shared_ptr <MapOptimizerBase> Ptr;

        MapOptimizerBase(MapBase::Ptr map, ViewerBase::Ptr viewer, BundleAdjustmentBase::Ptr bundleAdjustment,
                         int newFrameNumLimit);

        virtual ~MapOptimizerBase() = default;

        virtual void run();

        void requestFinish();

        bool isFinished();

        inline void notify() {
            //{std::scoped_lock lock(cvMutex_);}
            conditionVariable_.notify_one();
        }

    protected:
        /*
         * Remember to set and unset "selectedForBA" flags for frames and landmarks.
         */
        virtual double localBundleAdjustment();

        /*
         * return the EndTimeStamp, fill landmarks_ and keyframes_
         */
        virtual double selectVariableForBA() = 0;

        virtual void clearFlagAndOptData();

        virtual void cullingMap() = 0;

        bool finishRequested();

        void setFinish();

        inline void wait() {
            std::unique_lock lock(cvMutex_);
            conditionVariable_.wait(lock);
        }

        MapBase::Ptr map_;
        ViewerBase::Ptr viewer_;
        BundleAdjustmentBase::Ptr bundleAdjustment_;

        // run optimizer after "newFrameNumLimit_" new frames
        int newFrameNumLimit_;

        double lastFrameStamp_;

        bool finishRequested_;
        std::mutex mutexFinish_;
        bool finished_;

        std::map<double, Bodyframe::Ptr> keyframes_;
        std::map<int, LandmarkBase::Ptr> landmarks_;

        std::mutex cvMutex_;
        std::condition_variable conditionVariable_;
    };
}

#endif //OPENGV2_MAPOPTIMIZERBASE_HPP
