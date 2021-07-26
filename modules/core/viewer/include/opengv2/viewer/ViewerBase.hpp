//
// Created by huangkun on 2020/1/9.
//

#ifndef OPENGV2_VIEWERBASE_HPP
#define OPENGV2_VIEWERBASE_HPP

#include <shared_mutex>
#include <thread>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>

#include <opengv2/map/MapBase.hpp>
#include <opengv2/tracking/TrackingBase.hpp>

namespace opengv2 {
    class ViewerBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef std::shared_ptr<ViewerBase> Ptr;

        explicit ViewerBase(const std::string &strSettingPath, MapBase::Ptr map, TrackingBase::Ptr tracking);

        virtual ~ViewerBase() = default;

        // Main thread function
        virtual void run();

        void requestFinish();

        void requestStop();

        bool isFinished();

        bool isStopped();

        virtual void recovery();

        void drawCurrentFrame(pangolin::OpenGlMatrix &glTwc) const noexcept;

        void drawKeyFrames(bool drawCamera, bool drawConnection);

        virtual void drawLandmarks() const = 0;

        void drawGT(Bodyframe::Ptr bf);

        void drawComparison();

        inline void setGT(std::vector<std::map<double, Eigen::Matrix<double, 7, 1>>> *GT_Tws) {
            GT_Tws_ = GT_Tws;
        }

        /*
         * Align two trajectories using the method of Horn (closed-form).
         * model and data should be the same size
         */
        static void align(const std::vector<Eigen::Vector3d> &model, const std::vector<Eigen::Vector3d> &data,
                          Eigen::Quaterniond &Qdm, Eigen::Vector3d &tdm);

        /**
         * @brief update Viewer only when there are changes in framebuffer.
         */
        inline void updateViewer() {
            std::scoped_lock lock(updateMutex_);
            update_ = true;
        }

        /**
         * @brief for visualization of different result
         * save trajectory in std::vector<Eigen::Vector3d> for draw in Viewer
         */
        void saveTrajectory();

    protected:

        bool checkSystemState();

        virtual void updateData() = 0;

        inline void getCurrentOpenGLMatrix(Bodyframe::Ptr bf, pangolin::OpenGlMatrix &glTwc);

        virtual cv::Mat getCurrentImages(Bodyframe::Ptr bf, bool drawMatch) const = 0;

        float T_; // 1/fps in ms
        float viewpointX_, viewpointY_, viewpointZ_, viewpointF_;

        float cameraSize_, cameraLineWidth_;
        float keyFrameSize_, keyFrameLineWidth_, graphLineWidth_;

        bool stopped_;
        bool stopRequested_;
        std::mutex mutexStop_;

        bool finishRequested_;
        bool finished_;
        std::mutex mutexFinish_;

        int localRange_;
        bool showLocal_;

        Eigen::Matrix4d Transform_; // make the obj in viewer face forward

        bool stop();

        bool checkFinish();

        void setFinish();

        MapBase::Ptr map_;
        TrackingBase::Ptr tracking_;

        std::shared_mutex updateMutex_;
        bool update_;
        std::map<double, Bodyframe::Ptr> keyframes_;
        std::map<int, LandmarkBase::Ptr> landmarks_;

        std::vector<std::vector<Eigen::Vector3d>> savedTrajectory_;
        std::vector<std::map<double, Eigen::Matrix<double, 7, 1>>> *GT_Tws_;
    };
}

#endif //OPENGV2_VIEWERBASE_HPP
