//
// Created by huangkun on 2020/11/3.
//

#include <fstream>
#include <experimental/filesystem>

#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>

#include <opengv2/frame/Bodyframe.hpp>
#include <opengv2/system/SystemBase.hpp>
#include <opengv2/viewer/PointLandmarkViewer.hpp>

using namespace opengv2;
using namespace std;

int main(int argc, char **argv) {
    if (argc != 7) {
        cerr << endl
             << "Usage: ./visulization_eventCalib settingFilePath GTFilePath EventTrajectoryFile startTime endTime fps"
             << endl;
        return 1;
    }

    //Check settings file
    cv::FileStorage fsSettings(string(argv[1]), cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "Failed to open settings file at: " << string(argv[1]) << endl;
        exit(-1);
    }

    // map part
    MapBase::Ptr map = make_shared<MapBase>();
    // viewer part
    PointLandmarkViewer::Ptr viewer = make_shared<PointLandmarkViewer>(string(argv[1]), map, nullptr);
    // Extrinsic parameters, to be calibrated
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> Qsb;
    std::vector<Eigen::Vector3d> tsb;
    Qsb.emplace_back(1, 0, 0, 0);
    tsb.emplace_back(0, 0, 0);

    SystemBase SLAM(nullptr, map, viewer, nullptr, Qsb, tsb, nullptr, nullptr, string(argv[1]));

    // add landmarks
    bool isAsymmetric = true;
    fsSettings["Is_Pattern_Asymmetric"] >> isAsymmetric;
    int rows = fsSettings["BoardSize_Rows"];
    int cols = fsSettings["BoardSize_Cols"];
    double squareSize = fsSettings["Square_Size"];
    if (isAsymmetric) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                auto lm = std::make_shared<LandmarkBase>
                        (j + i * cols, Eigen::Vector3d((2 * j + i % 2) * squareSize, i * squareSize, 0));
                SLAM.map->addLandmark(lm);
            }
    } else {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) {
                auto lm = std::make_shared<LandmarkBase>
                        (j + i * cols, Eigen::Vector3d(j * squareSize, i * squareSize, 0));
                SLAM.map->addLandmark(lm);
            }
    }

    // load trajectory
    std::map<double, Eigen::Matrix<double, 7, 1>> GT_Tws, event_Tws;
    opengv2::SystemBase::loadTUM(string(argv[2]), GT_Tws);
    opengv2::SystemBase::loadTUM(string(argv[3]), event_Tws);

    // load image
    string path = string(argv[3]);
    auto index = path.find_last_of('/');
    string folder = path.substr(0, index);
    std::map<double, cv::Mat> eventImages, undistortedImages;
    for (const auto &entry : experimental::filesystem::directory_iterator(folder + "/image/")) {
        auto beginIdx = entry.path().string().find_last_of('/');
        auto endIdx = entry.path().string().find_last_of('.');
        string rawName = entry.path().string().substr(beginIdx + 1, endIdx - beginIdx - 1);
        double timestamp = std::stod(rawName);
        cv::Mat image = imread(entry.path().string(), cv::IMREAD_COLOR);
        eventImages.emplace(timestamp, image);
    }
    for (const auto &entry : experimental::filesystem::directory_iterator(folder + "/undistortedImage/")) {
        auto beginIdx = entry.path().string().find_last_of('/');
        auto endIdx = entry.path().string().find_last_of('_');
        if (entry.path().string()[endIdx + 1] == 'd')
            continue;
        string rawName = entry.path().string().substr(beginIdx + 1, endIdx - beginIdx - 1);
        double timestamp = std::stod(rawName);
        cv::Mat image = imread(entry.path().string(), cv::IMREAD_COLOR);
        undistortedImages.emplace(timestamp, image);
    }

    SLAM.loadGT(0, GT_Tws);
    cv::namedWindow("Current Frame");
    bool start = true;
    double fps = std::stod(argv[6]);
    std::pair duration(std::stod(argv[4]), std::stod(argv[5]));
    for (auto itr = eventImages.lower_bound(duration.first); itr != eventImages.lower_bound(duration.second); itr++) {
        auto eventItr = event_Tws.lower_bound(itr->first);
        if (eventItr == event_Tws.end()) {
            eventItr--;
        } else if (eventItr != event_Tws.begin()) {
            auto n2 = eventItr--;
            eventItr = std::abs(itr->first - eventItr->first) > std::abs(itr->first - n2->first) ? n2 : eventItr;
        }
        if (!SLAM.map->hasKeyframe(eventItr->first)) {
            Eigen::Quaterniond unitQws(eventItr->second.data());
            Eigen::Vector3d tws(eventItr->second.data() + 4);

            vector<FrameBase::Ptr> frames(1);
            auto bf = std::make_shared<Bodyframe>(frames, eventItr->first, tws, unitQws);
            SLAM.map->addFrame(bf);
        }

        auto undisItr = undistortedImages.lower_bound(itr->first);
        if (undisItr == undistortedImages.end()) {
            undisItr--;
        } else if (undisItr != undistortedImages.begin()) {
            auto n2 = undisItr--;
            undisItr = std::abs(itr->first - undisItr->first) > std::abs(itr->first - n2->first) ? n2 : undisItr;
        }
        cv::Mat undis;
        cv::resize(undisItr->second, undis, cv::Size(itr->second.size[1], itr->second.size[0]));

        cv::Mat dst;
        cv::Mat matArray[] = {undis, itr->second};
        cv::hconcat(matArray, 2, dst);
        cv::imshow("Current Frame", dst);
        cv::waitKey(1000 / fps);
        if (start) {
            cv::waitKey();
            start = false;
        }
    }

    std::cout << "press any key to exit..." << std::endl;
    std::cin.ignore();
    SLAM.shutdown();
    return 0;
}