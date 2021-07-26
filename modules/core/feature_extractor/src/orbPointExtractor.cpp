//
// Created by huangkun on 2020/6/2.
//

#include <opengv2/feature_extractor/orbPointExtractor.hpp>
#include <opengv2/feature/KeyPoint.hpp>

opengv2::orbPointExtractor::orbPointExtractor(std::shared_ptr <ORB_SLAM2::ORBextractor> feature2D) : feature2D_(
        feature2D) {}

bool opengv2::orbPointExtractor::extractFeature(const cv::Mat &img, const cv::Mat &mask,
                                                std::vector <FeatureBase::Ptr> &features) const {
    // TODO
}

void opengv2::orbPointExtractor::extractDescriptor(const cv::Mat &img, std::vector <FeatureBase::Ptr> &features,
                                                   cv::Mat &descriptors) const {
    // TODO
}

void
opengv2::orbPointExtractor::extractFeatureAndDescriptor(const cv::Mat &img, const cv::Mat &mask,
                                                        std::vector <FeatureBase::Ptr> &features,
                                                        cv::Mat &descriptors) const {
    std::vector<cv::KeyPoint> cvKeyPoints;

    feature2D_->operator()(img, mask, cvKeyPoints, descriptors);

    features.clear();
    for (auto &it:cvKeyPoints) {
        KeyPoint::Ptr kp = std::make_shared<KeyPoint>(it);
        features.push_back(kp);
    }
}