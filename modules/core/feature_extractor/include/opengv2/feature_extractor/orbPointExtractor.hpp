//
// Created by huangkun on 2020/6/2.
//

#ifndef OPENGV2_ORBPOINTEXTRACTOR_HPP
#define OPENGV2_ORBPOINTEXTRACTOR_HPP

#include "ORBextractor.h"

#include <opengv2/feature_extractor/FeatureExtractorBase.hpp>

namespace opengv2 {
    class orbPointExtractor : public FeatureExtractorBase {
    public:
        explicit orbPointExtractor(std::shared_ptr<ORB_SLAM2::ORBextractor> feature2D);

        bool
        extractFeature(const cv::Mat &img, const cv::Mat &mask, std::vector<FeatureBase::Ptr> &features) const override;

        void extractDescriptor(const cv::Mat &img, std::vector<FeatureBase::Ptr> &features,
                               cv::Mat &descriptors) const override;

        void
        extractFeatureAndDescriptor(const cv::Mat &img, const cv::Mat &mask, std::vector<FeatureBase::Ptr> &features,
                                    cv::Mat &descriptors) const override;

    protected:
        std::shared_ptr<ORB_SLAM2::ORBextractor> feature2D_;
    };
}

#endif //OPENGV2_ORBPOINTEXTRACTOR_HPP
