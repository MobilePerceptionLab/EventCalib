//
// Created by huangkun on 2020/8/11.
//

#include <opengv2/feature_extractor/cvPointExtractor.hpp>
#include <opengv2/feature/KeyPoint.hpp>

opengv2::cvPointExtractor::cvPointExtractor(cv::Ptr <cv::Feature2D> feature2D) : feature2D_(feature2D) {}

bool opengv2::cvPointExtractor::extractFeature(const cv::Mat &img, const cv::Mat &mask,
                                               std::vector <FeatureBase::Ptr> &features) const {
    std::vector <cv::KeyPoint> cvKeyPoints;
    feature2D_->detect(img, cvKeyPoints, mask);

    for (auto &it:cvKeyPoints) {
        KeyPoint::Ptr kp = std::make_shared<KeyPoint>(it);
        features.push_back(kp);
    }
    return true;
}

void opengv2::cvPointExtractor::extractDescriptor(const cv::Mat &img, std::vector <FeatureBase::Ptr> &features,
                                                  cv::Mat &descriptors) const {
    std::vector <cv::KeyPoint> cvKeyPoints;
    for (const FeatureBase::Ptr &it: features) {
        const auto *kp = dynamic_cast<const KeyPoint *>(it.get());
        if (kp == nullptr) {
            throw std::bad_cast();
        }
        cvKeyPoints.push_back(kp->cvKeyPoint);
    }

    feature2D_->compute(img, cvKeyPoints, descriptors);

    features.clear();
    for (auto &it:cvKeyPoints) {
        KeyPoint::Ptr kp = std::make_shared<KeyPoint>(it);
        features.push_back(kp);
    }
}

void
opengv2::cvPointExtractor::extractFeatureAndDescriptor(const cv::Mat &img, const cv::Mat &mask,
                                                       std::vector <FeatureBase::Ptr> &features,
                                                       cv::Mat &descriptors) const {
    std::vector<cv::KeyPoint> cvKeyPoints;
    feature2D_->detectAndCompute(img, mask, cvKeyPoints, descriptors);

    features.clear();
    for (auto &it:cvKeyPoints) {
        KeyPoint::Ptr kp = std::make_shared<KeyPoint>(it);
        features.push_back(kp);
    }
}