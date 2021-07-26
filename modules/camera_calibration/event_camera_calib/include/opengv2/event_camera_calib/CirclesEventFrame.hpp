//
// Created by huangkun on 2020/9/18.
//

#ifndef OPENGV2_CIRCLESEVENTFRAME_HPP
#define OPENGV2_CIRCLESEVENTFRAME_HPP

#include <nanoflann.hpp>
#include <KDTreeVectorOfVectorsAdaptor.h>

#include <opengv2/event/EventFrame.hpp>
#include <opengv2/event_camera_calib/parameters.hpp>
#include <opengv2/event_camera_calib/CalibCircle.hpp>
#include <opengv2/utility/utility.hpp>

//#include <cereal/types/base_class.hpp>
//#include <cereal/types/memory.hpp>
//#include <cereal/types/vector.hpp>

namespace opengv2 {
    class CirclesEventFrame : public EventFrame {
    public:
        struct Params {
            Params();

            explicit Params(const cv::FileStorage &node);

            double dbscan_eps; // pixel unit
            int dbscan_startMinSample;
            int clusterMinSample; // related with total event num.
            int knn_num; // k nearest neighbor
            bool fitCircle;
        };

        CirclesEventFrame(EventContainer::Ptr container, const std::pair<double, double> &duration,
                          CirclePatternParameters::Ptr pattern, Params params = Params());

        /**
         * @brief make sure in duration there are enough samples for each half circle.
         */
        bool extractFeatures();

        bool rectifyFeatures(const std::unordered_set<int> &outlierIdxs,
                             const Eigen::Ref<const Eigen::Matrix3d> &Rcw,
                             const Eigen::Ref<const Eigen::Vector3d> &tcw) override;

        /*
         * the returned lmRadius is just a zero centered circle point.
         */
        inline LandmarkBase::Ptr
        findCenter(const Eigen::Vector2d &p) const {
            size_t idx;
            double sqrDistance;
            circleKdTree_->query(p.data(), 1, &idx, &sqrDistance);

            double dis = std::sqrt(sqrDistance);
            auto f = dynamic_cast<CalibCircle *>(features_[idx].get());
            double radius = f->radius;

            if (std::abs(dis - radius) < 5) { // pixel unit
                return f->landmark();
            } else {
                return nullptr;
            }
        }

        // DEBUG
        //cv::Mat circleExtractionImage, detectionImage;
        //cv::Mat eventImage, clusterImage;

    protected:
        /*
         * Minimize geometric distance : \cite{A circle fitting procedure and its error analysis}
         * TODO: find an eclipse fitting algorithm.
         */
        void fitCircle(const std::vector<uint> &pSet,
                       const std::vector<uint> &nSet,
                       Eigen::Ref<Eigen::Vector2d> center, double &radius);

        CirclePatternParameters::Ptr pattern_;
        Params params_;

        double circleRadiusThreshold_;

        // the raw clustering result
        std::vector<std::vector<uint>> nClusters_, pClusters_;

        // features_: store the estimated circle center and radius, need to be rectified after opencv calibration.

        // KD-Tree for neighbor searching on features_, used for establish corresponds for given event.
        // TODO: considering ellipse
        std::shared_ptr<KDTreeVectorOfVectorsAdaptor<vectorofEigenMatrix<Eigen::Vector2d>, double, 2>> circleKdTree_;
        vectorofEigenMatrix<Eigen::Vector2d> circles_;
    };
}

#endif //OPENGV2_CIRCLESEVENTFRAME_HPP
