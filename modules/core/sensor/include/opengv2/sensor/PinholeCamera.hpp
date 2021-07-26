//
// Created by huangkun on 2019/12/30.
//

#ifndef OPENGV2_PINHOLECAMERA_HPP
#define OPENGV2_PINHOLECAMERA_HPP

//#include <cereal/types/base_class.hpp>

#include <opengv2/sensor/CameraBase.hpp>
#include <opengv2/utility/utility.hpp>

namespace opengv2 {
    class PinholeCamera : public CameraBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PinholeCamera(const Eigen::Ref<const Eigen::Vector2d> &size, const Eigen::Ref<const Eigen::Matrix3d> &K,
                      const Eigen::Ref<const Eigen::VectorXd> &distCoeffs = Eigen::VectorXd(),
                      cv::Mat mask = cv::Mat());

        explicit PinholeCamera(const cv::FileNode &sensorNode);

        inline const Eigen::Matrix3d &K() const noexcept {
            return K_;
        }

        inline const Eigen::Matrix3d &invK() const noexcept {
            return invK_;
        }

        inline Eigen::VectorXd &distCoeffs() noexcept {
            return distCoeffs_;
        }

        inline Eigen::VectorXd &inverseRadialPoly() noexcept {
            return inverseRadialPoly_;
        }

        inline Eigen::Vector2d project(const Eigen::Ref<const Eigen::Vector3d> &Xc) const override;

        inline Eigen::Vector3d invProject(const Eigen::Ref<const Eigen::Vector2d> &p) const override;

        inline void setK(const Eigen::Ref<const Eigen::Matrix3d> &cameraMatrix) {
            K_ = cameraMatrix;
            invK_ = K_.inverse();
        }

        /*
         * \cite{An Exact Formula for Calculating Inverse Radial Lens Distortions}
         * If the inverse not accurate in image bound use order = 9
         */
        static Eigen::VectorXd inverseRadialDistortion(const Eigen::Ref<const Eigen::Vector4d> &radialDistortion);

        inline Eigen::Vector2d undistortPoint(const Eigen::Ref<const Eigen::Vector2d> &p) const override;

        cv::Mat undistortImage(cv::Mat image) override;

        /*template<class Archive>
        void serialize(Archive &ar) {
            // We pass this cast to the base type for each base type we
            // need to serialize.  Do this instead of calling serialize functions
            // directly
            ar(cereal::base_class<CameraBase>(this), K_, invK_, distCoeffs_, inverseRadialPoly_);
        }*/

    protected:
        Eigen::Matrix3d K_;
        Eigen::Matrix3d invK_;

        // Opencv distortion coefficients (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements.
        Eigen::VectorXd distCoeffs_;

        // Inverse radial distortion polynomial, if set then will override opencv distortion when invProject.
        Eigen::VectorXd inverseRadialPoly_;

        // map corrected pixel coordinate (row, col) to weighted sum of distorted image pixel
        std::vector<std::vector<std::pair<std::vector<std::pair<int, int>>, std::vector<double>> >> undistortMap_;

        // opencv initUndistortRectifyMap
        cv::Mat map1_, map2_;
    };
}

#endif //OPENGV2_PINHOLECAMERA_HPP
