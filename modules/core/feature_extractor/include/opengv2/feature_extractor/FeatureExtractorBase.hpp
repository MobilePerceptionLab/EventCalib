//
// Created by huangkun on 2020/8/11.
//

#ifndef OPENGV2_FEATUREEXTRACTORBASE_HPP
#define OPENGV2_FEATUREEXTRACTORBASE_HPP

#include <opencv2/opencv.hpp>

#include <opengv2/feature/FeatureBase.hpp>

namespace opengv2 {
    class FeatureExtractorBase {
    public:
        typedef std::shared_ptr<FeatureExtractorBase> Ptr;

        virtual bool
        extractFeature(const cv::Mat &img, const cv::Mat &mask, std::vector<FeatureBase::Ptr> &features) const = 0;

        // Sometimes new features can be added, for example:
        // SIFT duplicates keypoint with several dominant orientations (for each orientation).
        virtual void
        extractDescriptor(const cv::Mat &img, std::vector<FeatureBase::Ptr> &features, cv::Mat &descriptors) const = 0;

        virtual void extractFeatureAndDescriptor(const cv::Mat &img, const cv::Mat &mask, std::vector<FeatureBase::Ptr> &features,
                                                 cv::Mat &descriptors) const = 0;
    };
}

#endif //OPENGV2_FEATUREEXTRACTORBASE_HPP
