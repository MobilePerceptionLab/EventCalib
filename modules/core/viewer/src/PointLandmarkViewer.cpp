//
// Created by huangkun on 2020/8/12.
//

#include <opencv2/opencv.hpp>

#include <opengv2/viewer/PointLandmarkViewer.hpp>
#include <opengv2/frame/CameraFrame.hpp>

opengv2::PointLandmarkViewer::PointLandmarkViewer(const std::string &strSettingPath, MapBase::Ptr map,
                                                  TrackingBase::Ptr tracking)
        : ViewerBase(strSettingPath, std::move(map), std::move(tracking)) {
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    pointSize_ = fSettings["Viewer.PointSize"];
}

void opengv2::PointLandmarkViewer::updateData() {
    if (tracking_ != nullptr) {
        std::scoped_lock lock(tracking_->matchMutex);
        if (!tracking_->lastMatchedFeatures.empty() &&
            !tracking_->laseMatches.empty()) {
            lastMatchedFeatures_ = std::move(tracking_->lastMatchedFeatures);
            laseMatches_ = std::move(tracking_->laseMatches);
        }
    }
}

cv::Mat opengv2::PointLandmarkViewer::getCurrentImages(Bodyframe::Ptr bf, bool drawMatch) const {
    if (bf != nullptr) {
        std::vector<cv::Mat> imgs;
        for (size_t i = 0; i < bf->size(); i++) {
            if (CameraFrame * cf = dynamic_cast<CameraFrame *>(bf->frame(i).get())) {
                if (cf->image().empty()) {
                    return cv::Mat();
                }

                imgs.push_back(cf->image());
                // Release image since no longer used
                cf->releaseImage();

                if (imgs.back().channels() < 3)
                    cvtColor(imgs.back(), imgs.back(), cv::COLOR_GRAY2BGR);

                if (drawMatch)
                    drawMatches(imgs.back(), i);
                else
                    drawMatchedFeatures(imgs.back(), i);
            }
        }

        if (imgs.empty())
            return cv::Mat();

        // w - Maximum number of images in a row
        // h - Maximum number of images in a column
        int w, h;

        int widthStep = imgs[0].cols, heightStep = imgs[0].rows;

        // If the number of arguments is lesser than 0 or greater than 12
        // return without displaying
        if (imgs.size() > 12) {
            throw std::logic_error(
                    "Number of arguments too large, can only handle maximally 12 images at a time ...\n");
        }

            // Determine the size of the image, and the number of rows/cols from number of arguments
        else if (imgs.size() == 1) {
            w = h = 1;
        } else if (imgs.size() == 2) {
            w = 1;
            h = 2;
        } else if (imgs.size() == 3 || imgs.size() == 4) {
            w = 2;
            h = 2;
        } else if (imgs.size() == 5 || imgs.size() == 6) {
            w = 2;
            h = 3;
        } else if (imgs.size() == 7 || imgs.size() == 8) {
            w = 2;
            h = 4;
        } else {
            w = 3;
            h = 4;
        }

        // Create a new image
        cv::Mat dispImage = cv::Mat::zeros(cv::Size(widthStep * w, heightStep * h), imgs[0].type());

        int k = 0;
        bool stopFlag = false;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (k < imgs.size()) {
                    if (imgs[k].empty()) {
                        throw std::logic_error("Image is empty");
                    }

                    cv::Rect ROI(x * widthStep, y * heightStep, widthStep, heightStep);
                    imgs[k].copyTo(dispImage(ROI));

                    k++;
                } else {
                    stopFlag = true;
                    break;
                }
            }
            if (stopFlag)
                break;
        }

        return dispImage;
    } else {
        return cv::Mat();
    }
}

void opengv2::PointLandmarkViewer::drawMatches(cv::Mat &img, int frameId) const {
    for (const auto &m: laseMatches_) {
        if (m->matchType() == MATCH2D2D) {
            const auto match = dynamic_cast<const Match2D2D *>(m.get());
            const auto &fi1 = match->fi1();
            const auto &fi2 = match->fi2();
            if (fi1.frameId() == frameId) {
                const FeatureBase::Ptr f1 = fi1.feature();
                const FeatureBase::Ptr f2 = fi2.feature();
                if (f1 == nullptr || f2 == nullptr)
                    continue;
                cv::Point2f kp1(f1->location()[0], f1->location()[1]);
                cv::Point2f kp2(f2->location()[0], f2->location()[1]);

                cv::line(img, kp1, kp2, cv::Scalar(0, 255, 0));
            }
        }
    }
}

void opengv2::PointLandmarkViewer::drawMatchedFeatures(cv::Mat &img, int frameId) const {
    const float r = 5;
    for (const FeatureIdentifier &fi: lastMatchedFeatures_) {
        if (fi.frameId() == frameId) {
            const FeatureBase::Ptr feature = fi.feature();
            if (feature == nullptr)
                continue;

            cv::Point2f kp(feature->location()[0], feature->location()[1]);

            cv::Point2f pt1, pt2;
            pt1.x = kp.x - r;
            pt1.y = kp.y - r;
            pt2.x = kp.x + r;
            pt2.y = kp.y + r;

            cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0));
            cv::circle(img, kp, 2, cv::Scalar(0, 255, 0), -1);
        }
    }
}

void opengv2::PointLandmarkViewer::drawLandmarks() const {
    std::vector<LandmarkBase::Ptr> currentLandmarks;
    Bodyframe::Ptr currentFrame = keyframes_.crbegin()->second;

    glPointSize(pointSize_);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0);
    for (const auto &itr: landmarks_) {
        auto &lm = itr.second;
        if (!lm->emptyObservations()) {
            if (currentFrame != nullptr &&
                lm->lastObservation().timestamp() == currentFrame->timeStamp()) {
                currentLandmarks.push_back(lm);
                continue;
            }
            glVertex3d(lm->position()(0), lm->position()(1), lm->position()(2));
        }
    }
    glEnd();

    glPointSize(pointSize_);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);
    for (const LandmarkBase::Ptr &lm: currentLandmarks) {
        if (currentFrame != nullptr && !lm->emptyObservations() &&
            lm->lastObservation().timestamp() == currentFrame->timeStamp()) {
            glVertex3d(lm->position()(0), lm->position()(1), lm->position()(2));
        }
    }
    glEnd();
}