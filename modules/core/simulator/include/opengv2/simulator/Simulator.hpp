//
// Created by huangkun on 2020/7/25.
//

#ifndef OPENGV2_SIMULATOR_HPP
#define OPENGV2_SIMULATOR_HPP

#include <random>

#include <opengv2/map/MapBase.hpp>

namespace opengv2 {
    class Simulator {
    public:
        explicit Simulator(SensorBase::Ptr sensor, bool randomSeed = true);

        void generateTrajectory(const MapBase::Ptr &map, double scale = 0.1);

        static void clearLandmarks(const MapBase::Ptr &map);

        void generateLandmarks(const MapBase::Ptr &map, int localConnectivity = 20);

        void generateObs(const MapBase::Ptr &map, int globalConnectivity = 1);

        void addNoise(const MapBase::Ptr &map, int noiseLevel, double proportion = 1.0);

    protected:
        std::mt19937 gen_; // random generator
        SensorBase::Ptr sensor_;
    };
}

#endif //OPENGV2_SIMULATOR_HPP
