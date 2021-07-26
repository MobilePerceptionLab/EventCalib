//
// Created by huangkun on 2020/9/18.
//

#ifndef OPENGV2_EVENTCONTAINER_HPP
#define OPENGV2_EVENTCONTAINER_HPP

#include <Eigen/Eigen>
#include <map>
#include <memory>

#include <opengv2/sensor/CameraBase.hpp>

namespace opengv2 {
    struct Event_loc_pol {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Event_loc_pol(const Eigen::Ref<const Eigen::Vector2d> &loc, bool polarity) : location(loc),
                                                                                     polarity(polarity) {}

        const Eigen::Vector2d location;
        const bool polarity;
    };

    struct EventContainer {
        typedef std::shared_ptr<EventContainer> Ptr;

        std::multimap<double, Event_loc_pol> container;
        CameraBase::Ptr camera;
    };
}

#endif //OPENGV2_EVENTCONTAINER_HPP
