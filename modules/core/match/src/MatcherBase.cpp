//
// Created by huangkun on 2020/2/15.
//

#include <opengv2/feature/FeatureIdentifier.hpp>
#include <opengv2/match/MatcherBase.hpp>

opengv2::MatcherBase::MatcherBase(cv::Ptr <cv::DescriptorMatcher> descriptorMatcher) :
        _descriptorMatcher(std::move(descriptorMatcher)) {}

opengv2::PointMatcher::PointMatcher(cv::Ptr <cv::DescriptorMatcher> descriptorMatcher)
        : MatcherBase(std::move(descriptorMatcher)) {}

void opengv2::PointMatcher::match2D2D(Bodyframe::Ptr bf1, Bodyframe::Ptr bf2, std::vector <Match2D2D::Ptr> &matches,
                                      double maxDisparity) const {
    matches.clear();
    for (int i = 0; i < bf1->size(); ++i) {
        if (CameraFrame * cf1 = dynamic_cast<CameraFrame *>(bf1->frame(i).get())) {
            CameraFrame *cf2 = dynamic_cast<CameraFrame *>(bf2->frame(i).get());
            std::vector <Match2D2D::Ptr> partMatches;
            match2D2D(cf1, cf2, bf1->timeStamp(), bf2->timeStamp(), i, partMatches, maxDisparity);
            std::move(partMatches.begin(), partMatches.end(), std::back_inserter(matches));
        }
    }
}

void opengv2::PointMatcher::match2D2D(CameraFrame *cf1, CameraFrame *cf2, double frameTime1, double frameTime2,
                                      int frameId, std::vector <MatchBase::Ptr> &matches, double maxDisparity) const {
    std::vector <cv::DMatch> cvMatches;
    // FLANN based matcher only support CV_32F
    auto d1 = cf1->descriptors();
    auto d2 = cf2->descriptors();
    if (d1.type() != CV_32F) {
        d1.convertTo(d1, CV_32F);
    }
    if (d2.type() != CV_32F) {
        d2.convertTo(d2, CV_32F);
    }
    _descriptorMatcher->match(d1, d2, cvMatches);

    std::vector<cv::DMatch> good_matches;
    for (auto &match : cvMatches) {
        Eigen::Vector3d bv1 = cf1->features()[match.queryIdx]->bearingVector();
        Eigen::Vector3d bv2 = cf2->features()[match.trainIdx]->bearingVector();
        bv1.normalize();
        bv2.normalize();
        // TODO: for stereo, depth consistency check needed.
        double disparity = 1 - bv1.transpose() * bv2;
        if (disparity <= maxDisparity) {
            good_matches.push_back(match);
        }
    }

    for (auto &match : good_matches) {
        FeatureIdentifier fi1(frameTime1, frameId, match.queryIdx, cf1->features()[match.queryIdx]);
        FeatureIdentifier fi2(frameTime2, frameId, match.trainIdx, cf2->features()[match.trainIdx]);
        matches.push_back(std::make_shared<Match2D2D>(fi1, fi2, match.distance));
    }
}

void opengv2::PointMatcher::match2D3D(Bodyframe::Ptr bf1, const std::vector <LandmarkBase::Ptr> &lm2,
                                      std::vector <Match2D3D::Ptr> &matches) const {
    // TODO
}