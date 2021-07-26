//
// Created by huangkun on 2020/8/12.
//

#ifndef OPENGV2_MATCHERBASE_HPP
#define OPENGV2_MATCHERBASE_HPP

#include <opencv2/opencv.hpp>

#include <opengv2/frame/Bodyframe.hpp>
#include <opengv2/frame/CameraFrame.hpp>
#include <opengv2/landmark/LandmarkBase.hpp>
#include <opengv2/match/MatchBase.hpp>

namespace opengv2 {
    class MatcherBase {
    public:
        typedef std::shared_ptr<MatcherBase> Ptr;

        explicit MatcherBase(cv::Ptr<cv::DescriptorMatcher> descriptorMatcher);

        // the disparity is for bearing vector, in 1-cos\theta form.
        virtual void match2D2D(Bodyframe::Ptr bf1, Bodyframe::Ptr bf2, std::vector<Match2D2D::Ptr> &matches,
                               double maxDisparity) const = 0;

        // the disparity is for bearing vector, in 1-cos\theta form.
        virtual void match2D2D(CameraFrame *cf1, CameraFrame *cf2, double frameTime1, double frameTime2,
                               int frameId, std::vector<MatchBase::Ptr> &matches, double maxDisparity) const = 0;

        virtual void match2D3D(Bodyframe::Ptr bf1, const std::vector<LandmarkBase::Ptr> &lm2,
                               std::vector<Match2D3D::Ptr> &matches) const = 0;

    protected:
        cv::Ptr<cv::DescriptorMatcher> _descriptorMatcher;
    };

    class PointMatcher : public MatcherBase {
    public:
        explicit PointMatcher(cv::Ptr<cv::DescriptorMatcher> descriptorMatcher);

        void match2D2D(Bodyframe::Ptr bf1, Bodyframe::Ptr bf2, std::vector<Match2D2D::Ptr> &matches,
                       double maxDisparity) const override;

        void match2D2D(CameraFrame *cf1, CameraFrame *cf2, double frameTime1, double frameTime2,
                       int frameId, std::vector<MatchBase::Ptr> &matches, double maxDisparity) const override;

        void match2D3D(Bodyframe::Ptr bf1, const std::vector<LandmarkBase::Ptr> &lm2,
                       std::vector<Match2D3D::Ptr> &matches) const override;
    };
}

#endif //OPENGV2_MATCHERBASE_HPP
