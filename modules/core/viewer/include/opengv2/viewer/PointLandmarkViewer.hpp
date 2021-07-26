//
// Created by huangkun on 2020/1/11.
//

#ifndef OPENGV2_POINTLANDMARKVIEWER_HPP
#define OPENGV2_POINTLANDMARKVIEWER_HPP

#include <opengv2/match/MatchBase.hpp>
#include <opengv2/viewer/ViewerBase.hpp>

namespace opengv2 {
    class PointLandmarkViewer : public ViewerBase {
    public:
        explicit PointLandmarkViewer(const std::string &strSettingPath, MapBase::Ptr map, TrackingBase::Ptr tracking);

        void drawLandmarks() const override;

        void drawMatchedFeatures(cv::Mat &img, int frameId) const;

        void drawMatches(cv::Mat &img, int frameId) const;

        cv::Mat getCurrentImages(Bodyframe::Ptr bf, bool drawMatch) const override; // TODO: Virtual: multi image with different size

        void updateData() override;

    protected:
        double pointSize_;

        std::vector<FeatureIdentifier> lastMatchedFeatures_;
        std::vector<MatchBase::Ptr> laseMatches_; // for debugging
    };
}

#endif //OPENGV2_POINTLANDMARKVIEWER_HPP
