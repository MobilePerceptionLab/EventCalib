//
// Created by huangkun on 2020/1/12.
//

#ifndef OPENGV2_SENSORBASE_HPP
#define OPENGV2_SENSORBASE_HPP

#include <memory>
//#include <cereal/types/base_class.hpp>

namespace opengv2 {
    enum SensorType {
        Camera, IMU
    }; // Note: Camera doesn't mean it's frame-based camera.

    class SensorBase {
    public:
        typedef std::shared_ptr<SensorBase> Ptr;

        explicit SensorBase(SensorType type) : type_(type) {};

        virtual ~SensorBase() = default;

        SensorBase &operator=(const SensorBase &other) = default;

        inline SensorType type() noexcept {
            return type_;
        }

        /*template<class Archive>
        void serialize(Archive &ar) { ar(type_); }*/

    protected:
        SensorType type_;
    };
}

#endif //OPENGV2_SENSORBASE_HPP
