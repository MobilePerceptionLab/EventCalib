//
// Created by huangkun on 2019/12/30.
//

#ifndef OPENGV2_CAMERABASE_HPP
#define OPENGV2_CAMERABASE_HPP

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
//#include <cereal/types/base_class.hpp>

#include <opengv2/sensor/SensorBase.hpp>
#include <opengv2/utility/utility.hpp>

namespace opengv2 {
    class CameraBase : public SensorBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        typedef std::shared_ptr<CameraBase> Ptr;

        explicit CameraBase(const Eigen::Ref<const Eigen::Vector2d> &size, cv::Mat mask = cv::Mat()) : SensorBase(
                SensorType::Camera), mask(mask), size_(size) {};

        explicit CameraBase(const cv::FileNode &sensorNode) : SensorBase(SensorType::Camera) {
            std::vector<double> v;
            cv::FileNode data = sensorNode["size"];
            if (!data.isSeq())
                throw std::invalid_argument(sensorNode.name() + ": size");
            for (cv::FileNodeIterator dataItr = data.begin(); dataItr != data.end(); dataItr++) {
                v.push_back(*dataItr);
            }
            if (v.size() != 2)
                throw std::invalid_argument(sensorNode.name() + ": size");
            size_ = Eigen::Vector2d(v.data());
            std::cout << "size: " << size_.transpose() << std::endl;

            data = sensorNode["mask"];
            if (!data.empty()) {
                if (!data.isString())
                    throw std::invalid_argument(sensorNode.name() + ": mask");
                mask = cv::imread(data.string(), cv::IMREAD_GRAYSCALE);
                std::cout << "mask: " << data.string() << std::endl;
            } else {
                mask = cv::Mat();
            }
        }

        inline const Eigen::Vector2d &size() const noexcept {
            return size_;
        }

        virtual inline Eigen::Vector2d project(const Eigen::Ref<const Eigen::Vector3d> &Xc) const = 0;

        // return unit vector
        virtual inline Eigen::Vector3d invProject(const Eigen::Ref<const Eigen::Vector2d> &p) const = 0;

        virtual inline Eigen::Vector2d undistortPoint(const Eigen::Ref<const Eigen::Vector2d> &p) const = 0;

        virtual cv::Mat undistortImage(cv::Mat image) = 0;

        /*template<class Archive>
        void serialize(Archive &ar) {
            // We pass this cast to the base type for each base type we
            // need to serialize.  Do this instead of calling serialize functions
            // directly
            ar(cereal::base_class<SensorBase>(this), size_);
        }*/

        cv::Mat mask;

    protected:
        // In opencv, src.at(i,j) is using (i,j) as (row,column) but Point(x,y) is using (x,y) as (column,row),
        // in which Point(x,y) is the same with Camera image coordinate (width, height)
        Eigen::Vector2d size_;
    };
}

#endif //OPENGV2_CAMERABASE_HPP
