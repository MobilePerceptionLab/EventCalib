//
// Created by huangkun on 2020/9/1.
//

#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include <opengv2/system/SystemBase.hpp>
#include <opengv2/bundle_adjustment/SplineBundleAdjustmentV2.hpp>
#include <opengv2/simulator/Simulator.hpp>
#include <opengv2/viewer/PointLandmarkViewer.hpp>
#include <opengv2/sensor/OmniScaramuzzaCamera.hpp>
#include <opengv2/spline/BsplineSO3.hpp>

using namespace std;
using namespace opengv2;

int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << endl
             << "Usage: ./simulation path_to_settings"
             << endl;
        return 1;
    }

    //Check settings file
    cv::FileStorage fsSettings(string(argv[1]), cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "Failed to open settings file at: " << string(argv[1]) << endl;
        exit(-1);
    }

    /**************************************** Create system ********************************/
    // map part
    MapBase::Ptr map = make_shared<MapBase>();
    // viewer part
    PointLandmarkViewer::Ptr viewer = make_shared<PointLandmarkViewer>(string(argv[1]), map, nullptr);
    // Extransic parameters
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> Qsb;
    std::vector<Eigen::Vector3d> tsb;
    double w = fsSettings["Qsb[0].w"];
    double x = fsSettings["Qsb[0].x"];
    double y = fsSettings["Qsb[0].y"];
    double z = fsSettings["Qsb[0].z"];
    Qsb.emplace_back(w, x, y, z);
    double tx = fsSettings["tsb[0].x"];
    double ty = fsSettings["tsb[0].y"];
    double tz = fsSettings["tsb[0].z"];
    tsb.emplace_back(tx, ty, tz);

    SystemBase SLAM(nullptr, map, viewer, nullptr, Qsb, tsb, nullptr, nullptr, string(argv[1]));

    /**************************************** Processing ********************************/
    // create sensor
    int width = fsSettings["Camera.width"];
    int height = fsSettings["Camera.height"];
    Eigen::Vector2d cs(width, height);
    double cx = fsSettings["Camera.cx"];
    double cy = fsSettings["Camera.cy"];
    Eigen::Vector2d principal_point(cx, cy);

    vector<double> v;
    cv::FileNode data = fsSettings["Camera.cam2world"];
    for (cv::FileNodeIterator itData = data.begin(); itData != data.end(); ++itData) {
        v.push_back(*itData);
    }
    Eigen::Matrix<double, 5, 1> polynomial(v.data());

    v.clear();
    data = fsSettings["Camera.world2cam"];
    for (cv::FileNodeIterator itData = data.begin(); itData != data.end(); ++itData) {
        v.push_back(*itData);
    }
    Eigen::Matrix<double, 12, 1> inverse_polynomial(v.data());

    v.clear();
    data = fsSettings["Camera.distortion"];
    for (cv::FileNodeIterator itData = data.begin(); itData != data.end(); ++itData) {
        v.push_back(*itData);
    }
    Eigen::Vector3d distortion(v.data());

    OmniScaramuzzaCamera::Ptr camera = make_shared<OmniScaramuzzaCamera>(cs, polynomial, principal_point, distortion,
                                                                         inverse_polynomial);

    Simulator trajG(camera);
    trajG.generateTrajectory(SLAM.map, 0.5);
    map->keyframeLockShared();
    std::map<double, Eigen::Matrix<double, 7, 1>> GT_Twc;
    for (const auto &itr: SLAM.map->keyframes()) {
        auto bf = itr.second;

        Eigen::Vector3d twb = bf->twb();
        Eigen::Quaterniond Qwb = bf->unitQwb();

        Eigen::Vector3d tbc = -(Bodyframe::unitQsb(0).conjugate() * Bodyframe::tsb(0));
        Eigen::Vector3d twc = Qwb * tbc + twb;
        Eigen::Quaterniond Qwc = Qwb * Bodyframe::unitQsb(0).conjugate();

        Eigen::Matrix<double, 7, 1> Tws;
        Tws.head(4) = Qwc.coeffs();
        Tws.tail(3) = twc;
        GT_Twc[bf->timeStamp()] = std::move(Tws);
    }
    map->keyframeUnlockShared();
    SLAM.loadGT(0, GT_Twc);

    /**************************************** Testing ********************************/
    std::vector<Sophus::SO3d, Eigen::aligned_allocator<Sophus::SO3d>> samples_R;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> samples_t;
    std::vector<double> u;
    SLAM.map->keyframeLockShared();
    for (const auto &it: SLAM.map->keyframes()) {
        Bodyframe::Ptr bf = it.second;
        Eigen::Quaterniond Qwb = bf->unitQwb();
        Eigen::Vector3d twb = bf->twb();

        Eigen::Quaterniond Qbw = Qwb.conjugate();

        samples_R.emplace_back(Qbw);
        samples_t.emplace_back(twb[0], twb[1], twb[2]);
        u.push_back(bf->timeStamp());
    }
    BsplineReal<3> spline_t(4, samples_t, std::floor(SLAM.map->keyframes().size() / 3.0), u);
    BsplineSO3 spline_R(4, samples_R, std::floor(SLAM.map->keyframes().size() / 3.0), u);

    // extract spline pose
    for (const auto &it: SLAM.map->keyframes()) {
        Bodyframe::Ptr bf = it.second;

        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> Ders;
        spline_t.evaluate(bf->timeStamp(), 0, Ders);
        Eigen::Vector3d twb(Ders[0]);

        Sophus::SO3d Rbw;
        std::vector<Eigen::Vector3d> unused;
        spline_R.evaluate(bf->timeStamp(), 0, Rbw, unused);
        Eigen::Quaterniond Qbw = Rbw.unit_quaternion();

        bf->optT.clear();
        std::copy(Qbw.coeffs().data(), Qbw.coeffs().data() + 4, std::back_inserter(bf->optT));
        std::copy(twb.data(), twb.data() + 3, std::back_inserter(bf->optT));
    }

    // switch
    for (const auto &it: map->keyframes()) {
        Bodyframe::Ptr bf = it.second;

        // opt -> real
        Eigen::Quaterniond Qwb = bf->unitQwb();
        Eigen::Vector3d twb = bf->twb();

        if (!bf->optT.empty()) {
            // opt -> real
            Eigen::Quaterniond Qbw(bf->optT.data());
            Qbw.normalize();
            bf->setPose(Eigen::Vector3d(bf->optT.data() + 4), Qbw.conjugate());
        }

        // real -> opt
        bf->optT.clear();
        Eigen::Quaterniond Qbw = Qwb.conjugate();
        std::copy(Qbw.coeffs().data(), Qbw.coeffs().data() + 4, std::back_inserter(bf->optT));
        std::copy(twb.data(), twb.data() + 3, std::back_inserter(bf->optT));
    }

    map->keyframeUnlockShared();

    std::cout << "press any key to exit..." << std::endl;
    std::cin.ignore();
    SLAM.shutdown();

    return 0;
}