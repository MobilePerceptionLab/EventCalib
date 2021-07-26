//
// Created by huangkun on 2020/1/13.
//

#ifndef OPENGV2_CAMERAFRAME_HPP
#define OPENGV2_CAMERAFRAME_HPP

#include <Eigen/StdVector>
#include <opencv2/opencv.hpp>

#include <opengv2/frame/FrameBase.hpp>
#include <opengv2/sensor/CameraBase.hpp>
#include <opengv2/feature/FeatureBase.hpp>
#include <opengv2/feature_extractor/FeatureExtractorBase.hpp>

namespace opengv2 {
    // Note: when using dynamic_cast, use check nullptr to valid instead of using sensor type.
    class CameraFrame : public FrameBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CameraFrame(cv::Mat image, const CameraBase::Ptr &camera) :
                FrameBase(camera), image_(image), releaseImageRequested_(false) {}

        virtual inline cv::Mat image() const noexcept {
            return image_;
        }

        virtual inline void requestReleaseImage() noexcept {
            releaseImageRequested_ = true;
        }

        /**
         * @brief Only used by view thread, work with requestReleaseImage().
         */
        virtual inline void releaseImage() noexcept {
            if (releaseImageRequested_)
                image_.release();
        }

        virtual inline const std::vector<FeatureBase::Ptr> &features() const noexcept {
            return features_;
        }

        /**
         * @todo avoid such usage.
         */
        virtual inline std::vector<FeatureBase::Ptr> &features() noexcept {
            return features_;
        }

        /**
         * @brief set features from outsides, only for initialization
         * @param features std::move() used
         */
        virtual inline void setFeatures(std::vector<FeatureBase::Ptr> &features) noexcept {
            features_ = std::move(features);
        }

        virtual inline bool extractFeatures(const FeatureExtractorBase &extractor) {
            if (image_.empty()) {
                throw std::invalid_argument("CameraFrame.image_ is empty.");
            }

            return extractor.extractFeature(image_, dynamic_cast<CameraBase *>(sensor_.get())->mask, features_);
        }

        /**
         * @brief Before usage, make sure the intrinsics is calibrated
         */
        virtual inline void extractBearingVectors() {
            if (features_.empty()) {
                throw std::invalid_argument("CameraFrame.features_ is empty!");
            }

            for (auto &it:features_) {
                it->setBearingVector(dynamic_cast<CameraBase *>(sensor_.get())->invProject(it->location()));
            }
        }

        inline cv::Mat descriptors() const noexcept {
            return descriptors_;
        }

        inline void extractDescriptors(const FeatureExtractorBase &extractor) {
            if (image_.empty()) {
                throw std::invalid_argument("MonoCameraFrame.img_ is empty!");
            }
            if (features_.empty()) {
                throw std::invalid_argument("MonoCameraFrame.features_ is empty!");
            }

            extractor.extractDescriptor(image_, features_, descriptors_);
        }

        virtual inline void extractFeaturesAndDescriptors(const FeatureExtractorBase &extractor) {
            if (image_.empty()) {
                throw std::invalid_argument("MonoCameraFrame.img_ is empty");
            }

            extractor.extractFeatureAndDescriptor(image_, dynamic_cast<CameraBase *>(sensor_.get())->mask, features_,
                                                  descriptors_);
        }

    protected:
        // OpenCV imread, imwrite and imshow work with the BGR order.
        cv::Mat image_; // Remember to Release if no longer used
        bool releaseImageRequested_;

        std::vector<FeatureBase::Ptr> features_; // dynamic_cast only work for reference and pointer

        // TODO: Find a way to assign suitable descriptor to landmark, release descriptors in Frame
        cv::Mat descriptors_;
    };
}

#endif //OPENGV2_CAMERAFRAME_HPP
