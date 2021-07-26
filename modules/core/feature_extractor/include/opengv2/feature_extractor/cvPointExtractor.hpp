//
// Created by huangkun on 2019/12/31.
//

#ifndef OPENGV2_CVPOINTEXTRACTOR_HPP
#define OPENGV2_CVPOINTEXTRACTOR_HPP

#include <opengv2/feature_extractor/FeatureExtractorBase.hpp>

namespace opengv2 {
    class cvPointExtractor : public FeatureExtractorBase {
    public:
        explicit cvPointExtractor(cv::Ptr <cv::Feature2D> feature2D);

        bool
        extractFeature(const cv::Mat &img, const cv::Mat &mask,
                       std::vector <FeatureBase::Ptr> &features) const override;

        void extractDescriptor(const cv::Mat &img, std::vector <FeatureBase::Ptr> &features,
                               cv::Mat &descriptors) const override;

        void
        extractFeatureAndDescriptor(const cv::Mat &img, const cv::Mat &mask, std::vector<FeatureBase::Ptr> &features,
                                    cv::Mat &descriptors) const override;

    protected:
        cv::Ptr<cv::Feature2D> feature2D_;
    };
}

#endif //OPENGV2_CVPOINTEXTRACTOR_HPP
