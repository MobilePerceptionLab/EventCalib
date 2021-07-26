//
// Created by huangkun on 2020/11/3.
//

#include <fstream>

#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>

#include <opengv2/frame/Bodyframe.hpp>
#include <opengv2/system/SystemBase.hpp>
#include <opengv2/viewer/PointLandmarkViewer.hpp>

using namespace opengv2;
using namespace std;

void
loadGT(const string &file, long long baseTime, double shiftTime, std::map<double, Eigen::Matrix<double, 7, 1>> &gTwb);

void calibrateHandEye(const std::map<double, Eigen::Matrix<double, 7, 1>> &Tws,
                      const std::map<double, Eigen::Matrix<double, 7, 1>> &Tw1b,
                      Eigen::Quaterniond &unitQbs, Eigen::Vector3d &tbs);


int main(int argc, char **argv) {
    if (argc != 7) {
        cerr << endl
             << "Usage: ./handEyeCalib settingFilePath GTFilePath baseTime EventTrajectoryFile cvTrajectoryFile SavePath"
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

    // load GT, EventTrajectory cvTrajectory
    std::map<double, Eigen::Matrix<double, 7, 1>> GT_Twb, event_Tws, cv_Tws;
    double shiftTime = fsSettings["shiftTime"];
    loadGT(string(argv[2]), stoll(argv[3]), shiftTime, GT_Twb);
    opengv2::SystemBase::loadTUM(string(argv[4]), event_Tws);
    opengv2::SystemBase::loadTUM(string(argv[5]), cv_Tws);
    std::cout << event_Tws.size() << std::endl;
    std::cout << cv_Tws.size() << std::endl;

    // hand-eye calibration
    bool calculateTbs = true;
    fsSettings["calculateTbs"] >> calculateTbs;
    Eigen::Quaterniond unitQbs;
    Eigen::Vector3d tbs;
    if (calculateTbs) {
        std::cout << "calibrateHandEye for Event" << std::endl;
        Eigen::Quaterniond event_unitQbs;
        Eigen::Vector3d event_tbs;
        calibrateHandEye(event_Tws, GT_Twb, event_unitQbs, event_tbs);
        Eigen::Quaterniond cv_unitQbs;
        Eigen::Vector3d cv_tbs;
        if (cv_Tws.size() >= 4) {
            std::cout << "calibrateHandEye for CV" << std::endl;
            calibrateHandEye(cv_Tws, GT_Twb, cv_unitQbs, cv_tbs);
        }

        bool chooseCV = false;//event_tbs.norm() > cv_tbs.norm();
        std::cout << "Choose: " << (chooseCV ? "CV hand-eye result." : "Event hand-eye result.")
                  << std::endl;
        if (chooseCV) {
            unitQbs = cv_unitQbs;
            tbs = cv_tbs;
        } else {
            unitQbs = event_unitQbs;
            tbs = event_tbs;
        }
    } else {
        cv::FileNode data = fsSettings["Qbs"];
        std::vector<double> v;
        for (auto &&itData : data) {
            v.push_back(itData);
        }
        unitQbs = Eigen::Map<Eigen::Quaterniond>(v.data());
        unitQbs.normalize();

        data = fsSettings["tbs"];
        v.clear();
        for (auto &&itData : data) {
            v.push_back(itData);
        }
        tbs = Eigen::Map<Eigen::Vector3d>(v.data());
    }

    // save GT
    std::ofstream f(string(argv[6]) + "/gtTrajectory.txt", std::ofstream::trunc);
    f << std::fixed << std::setprecision(10);
    if (!f.is_open())
        throw std::logic_error("Unable to open files: " + string(argv[6]) + "/gtTrajectory.txt");
    for (auto &itr: GT_Twb) {//qw qx qy qz tx ty tz
        Eigen::Map<const Eigen::Quaterniond> unitQw1b(itr.second.data());
        Eigen::Map<const Eigen::Vector3d> tw1b(itr.second.data() + 4);

        Eigen::Quaterniond unitQw1s = unitQw1b * unitQbs;
        Eigen::Vector3d tw1s = unitQw1b * tbs + tw1b;
        itr.second.head<4>() = unitQw1s.coeffs();
        itr.second.tail<3>() = tw1s;

        //timestamp tx ty tz qx qy qz qw
        f << itr.first << " " << tw1s[0] << " " << tw1s[1] << " " << tw1s[2] << " "
          << unitQw1s.x() << " " << unitQw1s.y() << " " << unitQw1s.z() << " " << unitQw1s.w() << std::endl;
    }
    f.close();

    // TODO: add trajectory to map.
    for (const auto &itr: event_Tws) {
        Eigen::Quaterniond unitQws(itr.second.data());
        Eigen::Vector3d tws(itr.second.data() + 4);

        vector<FrameBase::Ptr> frames(1);
        auto bf = std::make_shared<Bodyframe>(frames, itr.first, tws, unitQws);
        SLAM.map->addFrame(bf);
    }
    SLAM.loadGT(0, GT_Twb);

    std::cout << "press any key to exit..." << std::endl;
    std::cin.ignore();
    SLAM.shutdown();
    return 0;
}

void calibrateHandEye(const std::map<double, Eigen::Matrix<double, 7, 1>> &Tws,
                      const std::map<double, Eigen::Matrix<double, 7, 1>> &Tw1b,
                      Eigen::Quaterniond &unitQbs, Eigen::Vector3d &tbs) {
    std::vector<cv::Mat> Rw1bSet, tw1bSet, RswSet, tswSet;
    for (const auto &itr: Tws) {
        // Tbw1Set
        auto n1 = Tw1b.lower_bound(itr.first);
        if (n1 == Tw1b.end()) {
            n1--;
        } else if (n1 != Tw1b.begin()) {
            auto n2 = n1--;
            n1 = std::abs(itr.first - n1->first) > std::abs(itr.first - n2->first) ? n2 : n1;
        }

        auto itr1 = Tws.lower_bound(n1->first);
        if (itr1 == Tws.end()) {
            itr1--;
        } else if (itr1 != Tws.begin()) {
            auto itr2 = itr1--;
            itr1 = std::abs(n1->first - itr1->first) > std::abs(n1->first - itr2->first) ? itr2 : itr1;
        }

        if (itr1->first != itr.first || std::abs(n1->first - itr.first) > 3e-2)
            continue;

        Eigen::Quaterniond unitQw1b(n1->second.data());
        Eigen::Vector3d tw1b(n1->second.data() + 4);
        cv::Mat cv_Rw1b, cv_tw1b;
        cv::eigen2cv(unitQw1b.toRotationMatrix(), cv_Rw1b);
        cv::eigen2cv(tw1b, cv_tw1b);
        Rw1bSet.push_back(cv_Rw1b);
        tw1bSet.push_back(cv_tw1b);

        // TswSet
        Eigen::Quaterniond unitQws(itr.second.data());
        Eigen::Vector3d tws(itr.second.data() + 4);
        cv::Mat cv_Rsw, cv_tsw;
        cv::eigen2cv(unitQws.conjugate().toRotationMatrix(), cv_Rsw);
        cv::eigen2cv(Eigen::Vector3d(unitQws.conjugate() * (-tws)), cv_tsw);
        RswSet.push_back(cv_Rsw);
        tswSet.push_back(cv_tsw);
    }

    cv::Mat cv_Rbs, cv_tbs;
    cv::calibrateHandEye(Rw1bSet, tw1bSet, RswSet, tswSet, cv_Rbs, cv_tbs,
                         cv::CALIB_HAND_EYE_PARK /*CALIB_HAND_EYE_HORAUD*/);
    Eigen::Matrix3d Rbs;
    cv::cv2eigen(cv_Rbs, Rbs);
    cv::cv2eigen(cv_tbs, tbs);
    unitQbs = Rbs;
    unitQbs.normalize();

    std::cout.precision(10);
    std::cout << "Estimated extrinsic parameter: " << std::endl <<
              unitQbs.coeffs().transpose() << std::endl << tbs.transpose() << std::endl;
}

void
loadGT(const string &file, long long baseTime, double shiftTime, std::map<double, Eigen::Matrix<double, 7, 1>> &gTwb) {
    //GT format:timestamp qx qy qz qw tx ty tz. diff from TUM
    std::ifstream fInput(file, std::ifstream::in);

    std::vector<double> T(8);
    long long time;
    char line[512];
    fInput.getline(line, 512);
    while (!fInput.eof() && !fInput.fail()) {
        std::stringstream ss(line, std::stringstream::in);
        ss >> time;
        T[0] = (time - baseTime) * 1e-6 + shiftTime;
        for (int i = 1; i < 8; ++i) {
            ss >> T[i];
        }

        Eigen::Quaterniond Qwb(T.data() + 1);
        Qwb.normalize(); // normalize for further usage
        Eigen::Matrix<double, 7, 1> Twb;
        Twb.head(4) = Qwb.coeffs();
        Twb.tail(3) << 100 * T[5], 100 * T[6], 100 * T[7]; // to cm
        gTwb[T[0]] = std::move(Twb);

        fInput.getline(line, 512);
    }
}