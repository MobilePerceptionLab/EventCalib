//
// Created by huangkun on 2020/7/25.
//

#ifndef OPENGV2_OMNIMEICAMERA_HPP
#define OPENGV2_OMNIMEICAMERA_HPP

#include <opengv2/sensor/CameraBase.hpp>
#include <cereal/types/base_class.hpp>
#include <opengv2/eigen_extra/eigen_extra.hpp>

namespace opengv2 {
    class OmniMeiCamera : public CameraBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        OmniMeiCamera(const Eigen::Ref<const Eigen::Vector2d> &size,
                      const Eigen::Ref<const Eigen::VectorXd> &intrinsic,
                      const Eigen::Ref<const Eigen::VectorXd> &distort);

        inline Eigen::Vector2d project(const Eigen::Ref<const Eigen::Vector3d> &Xc) const override;

        inline Eigen::Vector3d invProject(const Eigen::Ref<const Eigen::Vector2d> &p) const override;

        inline Eigen::Vector2d undistortPoint(const Eigen::Ref<const Eigen::Vector2d> &p) const override;

        cv::Mat undistortImage(cv::Mat image) override;

        Eigen::Vector2d distort(const Eigen::Ref<const Eigen::Vector2d> &pt) const;
        Eigen::Vector2d distort(const Eigen::Ref<const Eigen::Vector2d> &p, Eigen::Ref<Eigen::Matrix2d> J) const;

        template<class Archive>
        void serialize(Archive &ar) {
            // We pass this cast to the base type for each base type we
            // need to serialize.  Do this instead of calling serialize functions
            // directly
            ar(cereal::base_class<CameraBase>(this), intrinsic_, distort_);
        }

    protected:
        Eigen::VectorXd intrinsic_;
        Eigen::VectorXd distort_;
    };
}

#endif //OPENGV2_OMNIMEICAMERA_HPP
