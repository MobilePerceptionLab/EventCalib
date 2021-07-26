//
// Created by huangkun on 2020/1/11.
//

#ifndef OPENGV2_BUNDLEADJUSTMENTBASE_HPP
#define OPENGV2_BUNDLEADJUSTMENTBASE_HPP

#include <ceres/ceres.h>

#include <opengv2/map/MapBase.hpp>

namespace opengv2 {
    class BundleAdjustmentBase {
    public:
        typedef std::shared_ptr<BundleAdjustmentBase> Ptr;

        BundleAdjustmentBase() = default;

        virtual void
        run(const std::map<double, Bodyframe::Ptr> &keyframes, const std::map<int, LandmarkBase::Ptr> &landmarks) = 0;

    protected:
        virtual void optimize(ceres::Problem &problem,
                              const std::map<double, Bodyframe::Ptr> &keyframes,
                              const std::map<int, LandmarkBase::Ptr> &landmarks) = 0;
    };
}

#endif //OPENGV2_BUNDLEADJUSTMENTBASE_HPP
