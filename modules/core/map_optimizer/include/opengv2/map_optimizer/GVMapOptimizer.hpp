//
// Created by huangkun on 2021/6/17.
//

#ifndef OPENGV2_GVMAPOPTIMIZER_HPP
#define OPENGV2_GVMAPOPTIMIZER_HPP

#include <opengv2/map_optimizer/MapOptimizerBase.hpp>

namespace opengv2 {
    class GVMapOptimizer : public MapOptimizerBase {
    public:
        GVMapOptimizer(MapBase::Ptr map, ViewerBase::Ptr viewer, BundleAdjustmentBase::Ptr bundleAdjustment,
                       int newFrameNumLimit = 7);

    protected:
        double selectVariableForBA() override;

        // TODO: add support fro stereo
        void cullingMap() override;
    };
}

#endif //OPENGV2_GVMAPOPTIMIZER_HPP
