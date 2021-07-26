//
// Created by huangkun on 2020/7/25.
//

#ifndef OPENGV2_OMNISCARAMUZZACAMERA_HPP
#define OPENGV2_OMNISCARAMUZZACAMERA_HPP

// #include <cereal/types/base_class.hpp>

#include <opengv2/sensor/CameraBase.hpp>
#include <opengv2/utility/utility.hpp>

namespace opengv2 {
    class OmniScaramuzzaCamera : public CameraBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        OmniScaramuzzaCamera(const Eigen::Ref<const Eigen::Vector2d> &size,
                             const Eigen::Ref<const Eigen::VectorXd> &polynomial,
                             const Eigen::Ref<const Eigen::Vector2d> &principal_point,
                             const Eigen::Ref<const Eigen::Vector3d> &distortion,
                             const Eigen::Ref<const Eigen::VectorXd> &inverse_polynomial,
                             cv::Mat mask = cv::Mat());

        explicit OmniScaramuzzaCamera(const cv::FileNode &sensorNode);

        inline Eigen::Vector2d project(const Eigen::Ref<const Eigen::Vector3d> &Xc) const override;

        inline Eigen::Vector3d invProject(const Eigen::Ref<const Eigen::Vector2d> &p) const override;

        inline Eigen::Vector2d undistortPoint(const Eigen::Ref<const Eigen::Vector2d> &p) const override;

        cv::Mat undistortImage(cv::Mat image) override;

        /*template<class Archive>
        void serialize(Archive &ar) {
            // We pass this cast to the base type for each base type we
            // need to serialize.  Do this instead of calling serialize functions
            // directly
            ar(cereal::base_class<CameraBase>(this), polynomial_, principal_point_,
               inverse_polynomial_, affine_correction_, affine_correction_inverse_);
        }*/

    protected:
        inline Eigen::Matrix2d distortionToAffineCorrection(const Eigen::Ref<const Eigen::Vector3d> &distortion);

        Eigen::Vector2d principal_point_;

        // Important: we exchange x and y since our convention is to work with x along the columns and y along the rows
        // (different from Ocam convention)
        Eigen::VectorXd polynomial_;
        Eigen::VectorXd inverse_polynomial_;
        Eigen::Matrix2d affine_correction_;
        Eigen::Matrix2d affine_correction_inverse_;
    };
}

#endif //OPENGV2_OMNISCARAMUZZACAMERA_HPP
