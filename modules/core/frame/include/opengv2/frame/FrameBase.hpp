//
// Created by huangkun on 2019/12/30.
//

#ifndef OPENGV2_FRAMEBASE_HPP
#define OPENGV2_FRAMEBASE_HPP

#include <opengv2/sensor/SensorBase.hpp>

//#include <cereal/types/base_class.hpp>
//#include <cereal/types/memory.hpp>

namespace opengv2 {
    class FrameBase {
    public:
        typedef std::shared_ptr <FrameBase> Ptr;

        explicit FrameBase(SensorBase::Ptr sensor) : sensor_(sensor) {};

        virtual ~FrameBase() = default;

        FrameBase &operator=(const FrameBase &other) = default;

        inline SensorBase::Ptr sensor() const noexcept {
            return sensor_;
        }

        /*template<class Archive>
        void serialize(Archive &ar) { ar(sensor_); }*/

    protected:
        SensorBase::Ptr sensor_;
    };
}

#endif //OPENGV2_FRAMEBASE_HPP
