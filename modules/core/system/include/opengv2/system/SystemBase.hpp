//
// Created by huangkun on 2020/2/12.
//

#ifndef OPENGV2_SYSTEMBASE_HPP
#define OPENGV2_SYSTEMBASE_HPP

#include <Eigen/StdVector>

#include <thread>

#include <opengv2/viewer/ViewerBase.hpp>
#include <opengv2/tracking/TrackingBase.hpp>
#include <opengv2/map/MapBase.hpp>
#include <opengv2/match/MatcherBase.hpp>
#include <opengv2/feature_extractor/FeatureExtractorBase.hpp>
#include <opengv2/map_optimizer/MapOptimizerBase.hpp>

namespace opengv2 {
    class SystemBase {
    public:
        SystemBase(TrackingBase::Ptr tracking, MapBase::Ptr map, ViewerBase::Ptr viewer,
                   MapOptimizerBase::Ptr mapOptimizer,
                   std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> &unitQsb,
                   std::vector<Eigen::Vector3d> &tsb,
                   FeatureExtractorBase::Ptr featureExtractor, MatcherBase::Ptr matcher,
                   const std::string &strSettingsFile);

        // All threads will be requested to finish.
        // It waits until all threads have finished.
        // This function must be called before saving the trajectory.
        virtual void shutdown();

        // Save keyframe poses in the TUM RGB-D dataset format.
        // This method works for all sensor input.
        // Call first shutdown()
        // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
        void saveKeyFrameTrajectoryTUM(int sensorId, const std::string &filename);

        static void loadTUM(const std::string &file, std::map<double, Eigen::Matrix<double, 7, 1>> &T);

        // relative translational(ignore scale)/rotation error: mean and standard deviation
        // t_mean, t_stddev, R_mean, R_stddev
        void evaluation(int sensorId, std::vector<double> &result);

        void loadGT(int sensorId, std::map<double, Eigen::Matrix<double, 7, 1>> &GT_Tws);

        const std::map<double, Eigen::Matrix<double, 7, 1>> &GT_Tws(int sensorId);

        // apply GT scale to frames, landmark not changing
        void applyGTscale();

        // transform the 1st frame into identity for comparison
        void transformIdentity();

        TrackingBase::Ptr tracking;
        ViewerBase::Ptr viewer;
        MapOptimizerBase::Ptr mapOptimizer;
        MapBase::Ptr map;
        FeatureExtractorBase::Ptr featureExtractor;
        MatcherBase::Ptr matcher;

    protected:
        std::shared_ptr<std::thread> viewerThread_, mapOptimizerThread_;
        std::string strSettingsFile_;

        // <timestamp, qx qy qz qw (Eigen::Quaternion order, unit quaternion) x y z>
        std::vector<std::map<double, Eigen::Matrix<double, 7, 1>>> GT_Tws_;
    };
}

#endif //OPENGV2_SYSTEMBASE_HPP
