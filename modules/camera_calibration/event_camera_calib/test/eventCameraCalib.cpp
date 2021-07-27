//
// Created by huangkun on 2020/9/15.
//

#include <fstream>
#include <thread>
#include <experimental/filesystem>

#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>

#include <opengv2/event_camera_calib/EventCalibIni.hpp>
#include <opengv2/event_camera_calib/CirclesEventFrame.hpp>
#include <opengv2/event_camera_calib/parameters.hpp>
#include <opengv2/event/EventContainer.hpp>
#include <opengv2/event/EventStream.hpp>
#include <opengv2/sensor/PinholeCamera.hpp>
#include <opengv2/frame/Bodyframe.hpp>
#include <opengv2/system/SystemBase.hpp>
#include <opengv2/viewer/PointLandmarkViewer.hpp>
#include <opengv2/event_camera_calib/EventCalibSpline.hpp>

using namespace opengv2;
using namespace std;

/**
 * @brief multi thread processing on different time span
 */
class MultiProcess {
public:
    explicit MultiProcess(std::vector<std::pair<double, double>> &timeBoundSet) : timeBoundSet_(
            std::move(timeBoundSet)) {}

    void process(EventContainer::Ptr eventContainer, CirclePatternParameters::Ptr circlePatternParameters,
                 TrackingBase::Ptr tracking, CirclesEventFrame::Params &params, double motionTimeStep,
                 double len, double frameGap, int frameEventNumThreshold) {
        while (true) {
            std::pair<double, double> timeBound;
            timeBoundSetMutex_.lock();
            if (!timeBoundSet_.empty()) {
                timeBound = timeBoundSet_.back();
                timeBoundSet_.pop_back();
                timeBoundSetMutex_.unlock();
            } else {
                timeBoundSetMutex_.unlock();
                return;
            }

            std::pair<double, double> duration(timeBound.first, timeBound.first + len);
            while (duration.second < timeBound.second) {
                vector<FrameBase::Ptr> frames;
                auto frame = std::make_shared<CirclesEventFrame>(eventContainer, duration,
                                                                 circlePatternParameters, params);
                int events_num = frame->eventsNum();

                if (frame->extractFeatures()) {
                    frames.push_back(frame);
                    Bodyframe::Ptr bf = make_shared<Bodyframe>(frames, (duration.first + duration.second) / 2);

                    if (tracking->process(bf)) {
                        duration.first = duration.second + frameGap;
                        duration.second = duration.first + len;

                        // debug
                        // cv::waitKey();
                    } else {
                        if (events_num > frameEventNumThreshold || (duration.second - duration.first) > 3 * len) {
                            duration.first += motionTimeStep;
                            duration.second = duration.first + len;
                        } else {
                            duration.second += motionTimeStep;
                        }
                    }

                } else {
                    if (events_num > frameEventNumThreshold || (duration.second - duration.first) > 3 * len) {
                        duration.first += motionTimeStep;
                        duration.second = duration.first + len;
                    } else {
                        duration.second += motionTimeStep;
                    }

                    /*// DEBUG: Set 'threadNum = 1' and uncomment this code to tune parameters
                    std::cout << events_num << " events." << std::endl;
                    if (frame->eventImage.empty())
                        continue;
                    cv::namedWindow("Debug_check: EventImage");
                    cv::imshow("Debug_check: EventImage", frame->eventImage);
                    //if (frame->clusterImage.empty())
                    //    continue;
                    //cv::namedWindow("Debug_check: clusterImage");
                    //cv::imshow("Debug_check: clusterImage", frame->clusterImage);
                    cv::waitKey();*/
                }
            }
        }
    }

protected:
    std::shared_mutex timeBoundSetMutex_;
    std::vector<std::pair<double, double>> timeBoundSet_;
};

int main(int argc, char **argv) {
    if (argc != 4) {
        cerr << endl
             << "Usage: ./unit_test_eventCameraCalib settingFilePath binFilePath SavePath"
             << endl;
        return 1;
    }
    cout.precision(8);

    //Check settings file
    cv::FileStorage fsSettings(string(argv[1]), cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "Failed to open settings file at: " << string(argv[1]) << endl;
        exit(-1);
    }

    /**************************************** Create system ********************************/
    auto calibrationSetting = make_shared<CalibrationSetting>(fsSettings);

    // map part
    MapBase::Ptr map = make_shared<MapBase>();
    // tracking part
    double motionTimeStep = fsSettings["MotionTimeStep"];
    EventCalibIni::Ptr tracking = make_shared<EventCalibIni>(map, calibrationSetting, motionTimeStep);
    // viewer part
    ViewerBase::Ptr viewer = make_shared<PointLandmarkViewer>(string(argv[1]), map, tracking);
    // Extrinsic parameters, to be calibrated
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> Qsb;
    std::vector<Eigen::Vector3d> tsb;
    Qsb.emplace_back(1, 0, 0, 0);
    tsb.emplace_back(0, 0, 0);

    SystemBase SLAM(tracking, map, viewer, nullptr, Qsb, tsb, nullptr, nullptr, string(argv[1]));

    EventStream es((string(argv[2])));
    EventContainer::Ptr eventContainer = make_shared<EventContainer>();

    // create sensor
    int width = fsSettings["Camera.width"];
    int height = fsSettings["Camera.height"];
    Eigen::Vector2d cs(width, height);
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    Eigen::VectorXd distortion = Eigen::VectorXd::Zero(5);
    eventContainer->camera = make_shared<PinholeCamera>(cs, K, distortion);

    bool customEnd = !fsSettings["EndTime"].isNone();
    double startTime = fsSettings["StartTime"];
    double endTime;
    if (customEnd)
        endTime = fsSettings["EndTime"];
    while (!es.isEnd()) {
        if (customEnd && es.iterator()->timeStamp() >= endTime) {
            break;
        }
        if (es.iterator()->timeStamp() >= startTime) {
            eventContainer->container.emplace(es.iterator()->timeStamp(),
                                              Event_loc_pol(es.iterator()->location(), es.iterator()->polarity()));
        }
        es.iterator()++;
    }
    es.close();
    endTime = eventContainer->container.rbegin()->first;
    std::cout << "Events from " << startTime << " second to " << endTime << " second loaded." << std::endl;

    int frameEventNumThreshold = fsSettings["FrameEventNumThreshold"];
    double len = 3 * motionTimeStep;
    double frameGap = 5 * motionTimeStep;
    CirclesEventFrame::Params params(fsSettings);
    auto threadNum = std::thread::hardware_concurrency() - 2;
    auto pieceNum = 5 * threadNum;
    double step = (endTime - startTime) / pieceNum;
    std::vector<std::pair<double, double>> timeBoundSet;
    timeBoundSet.reserve(pieceNum);
    for (int k = 0; k < pieceNum; ++k) {
        timeBoundSet.emplace_back(endTime - step * (k + 1), endTime - step * k);
    }
    MultiProcess multiProcess(timeBoundSet);
    std::vector<std::thread> threads;
    threads.reserve(threadNum);
    for (int i = 0; i < threadNum; ++i) {
        threads.emplace_back(&MultiProcess::process, &multiProcess, eventContainer,
                             calibrationSetting->circlePatternParameters, SLAM.tracking,
                             std::ref(params), motionTimeStep, len, frameGap, frameEventNumThreshold);
    }
    for (auto &th: threads) {
        th.join();
    }

    SLAM.map->keyframeLockShared();
    std::cout << SLAM.map->keyframes().size() << " frames in Map." << std::endl;
    for (const auto &itr: SLAM.map->keyframes()) {
        auto cf = dynamic_cast<CirclesEventFrame *>(itr.second->frame(0).get());
        std::cout << "Frame " << itr.second->timeStamp() << " contain " << cf->eventsNum() << " events." << std::endl;
    }
    SLAM.map->keyframeUnlockShared();

    dynamic_cast<EventCalibIni *>(SLAM.tracking.get())->cvCalibration();

    std::cout << SLAM.map->frameNum() << " frames in Map after Initialization." << std::endl;

    bool useSO3 = false;
    bool reduceMap = false;
    fsSettings["useSO3"] >> useSO3;
    fsSettings["reduceMap"] >> reduceMap;
    EventCalibSpline eventCalibSpline(SLAM.map, eventContainer, useSO3, reduceMap, motionTimeStep,
                                      calibrationSetting->circlePatternParameters->circleRadius);
    SLAM.viewer->updateViewer();

    SLAM.saveKeyFrameTrajectoryTUM(0, string(argv[3]) + "/TrajectoryByEvent.txt");

    experimental::filesystem::remove_all(string(argv[3]) + "/image/");
    experimental::filesystem::create_directories(string(argv[3]) + "/image/");
    SLAM.map->keyframeLockShared();
    for (const auto &itr: SLAM.map->keyframes()) {
        auto cf = dynamic_cast<CirclesEventFrame *>(itr.second->frame(0).get());
        cv::imwrite(string(argv[3]) + "/image/" + std::to_string(itr.second->timeStamp()) + ".png",
                    cf->image());
        /*cv::imwrite(string(argv[3]) + "/image/" + std::to_string(itr.second->timeStamp()) + "_event.png",
                    cf->eventImage);
        cv::imwrite(string(argv[3]) + "/image/" + std::to_string(itr.second->timeStamp()) + "_extraction.png",
                    cf->circleExtractionImage);
        cv::imwrite(string(argv[3]) + "/image/" + std::to_string(itr.second->timeStamp()) + "_cluster.png",
                    cf->clusterImage);*/
    }
    SLAM.map->keyframeUnlockShared();

    std::cout << "press Enter to exit..." << std::endl;
    std::cin.ignore();
    SLAM.shutdown();
    return 0;
}