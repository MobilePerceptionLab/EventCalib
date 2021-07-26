//
// Created by huangkun on 2020/9/22.
//

#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>

#include <opengv2/event/EventFrame.hpp>

opengv2::EventFrame::EventFrame(opengv2::EventContainer::Ptr container, const std::pair<double, double> &duration)
        : CameraFrame(cv::Mat(), container->camera), container_(container), duration_(duration) {
    std::unordered_set<Eigen::Vector2d, EigenMatrixHash<Eigen::Vector2d>, std::equal_to<>,
            Eigen::aligned_allocator<Eigen::Vector2d>> positiveEvents, negativeEvents;
    for (auto itr = container_->container.lower_bound(duration_.first);
         itr != container_->container.upper_bound(duration_.second); itr++) {
        if (itr->second.polarity) {
            positiveEvents.insert(itr->second.location);
        } else {
            negativeEvents.insert(itr->second.location);
        }
    }

    // delete event, which have + and - event at the same pixel
    for (auto itr = positiveEvents.begin(); itr != positiveEvents.end();) {
        auto found = negativeEvents.find(*itr);
        if (found == negativeEvents.end()) {
            itr++;
        } else {
            negativeEvents.erase(found);
            itr = positiveEvents.erase(itr);
        }
    }

    std::move(positiveEvents.begin(), positiveEvents.end(), std::back_inserter(positiveEvents_));
    std::move(negativeEvents.begin(), negativeEvents.end(), std::back_inserter(negativeEvents_));
}

cv::Mat opengv2::EventFrame::undistortedImage(CameraBase::Ptr camera) const {
    cv::Mat image = cv::Mat(std::round(camera->size()[1] * 1.2), std::round(camera->size()[0] * 1.2), CV_8UC3,
                            cv::Vec3b(0, 0, 0));

    if (positiveEvents_.empty() || negativeEvents_.empty()) {
        throw std::logic_error("Events in Frame was cleared.");
    }

    for (const Eigen::Vector2d &p: positiveEvents_) {
        Eigen::Vector2d p_c = camera->undistortPoint(p);
        cv::Point loc(std::round(p_c[0] + camera->size()[0] * 0.1),
                      std::round(p_c[1] + camera->size()[1] * 0.1));
        image.at<cv::Vec3b>(loc) = cv::Vec3b(0, 0, 255);
    }
    for (const Eigen::Vector2d &p: negativeEvents_) {
        Eigen::Vector2d p_c = camera->undistortPoint(p);
        cv::Point loc(std::round(p_c[0] + camera->size()[0] * 0.1),
                      std::round(p_c[1] + camera->size()[1] * 0.1));
        image.at<cv::Vec3b>(loc) = cv::Vec3b(0, 255, 0);
    }

    return image;
}