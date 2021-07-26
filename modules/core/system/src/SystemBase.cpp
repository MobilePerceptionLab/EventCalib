//
// Created by huangkun on 2020/2/12.
//

#include <numeric>

#include <CmakeConfig.h>

#include <opengv2/system/SystemBase.hpp>

opengv2::SystemBase::SystemBase(TrackingBase::Ptr tracking, MapBase::Ptr map, ViewerBase::Ptr viewer,
                                MapOptimizerBase::Ptr mapOptimizer,
                                std::vector<Eigen::Quaterniond, Eigen::aligned_allocator < Eigen::Quaterniond>>

&unitQsb,
                                 std::vector<Eigen::Vector3d> &tsb,
                                 FeatureExtractorBase::Ptr featureExtractor, MatcherBase::Ptr matcher,
                                 const std::string &strSettingsFile) :
        tracking(tracking), map(map), featureExtractor(featureExtractor), matcher(matcher), viewer(viewer),
        mapOptimizer(mapOptimizer), strSettingsFile_(strSettingsFile) {
    GT_Tws_.resize(unitQsb.size());

    // extransic parameters
    Bodyframe::setTsb(unitQsb, tsb);

    if (tracking != nullptr) {
        tracking->setSystem(this);

        tracking->stateMutex.lock();
        tracking->state = NOT_INITIALIZED;
        tracking->stateMutex.unlock();
    }

    if (viewer != nullptr) {
        viewer->setGT(&GT_Tws_);
        viewerThread_ = std::make_shared<std::thread>(&ViewerBase::run, viewer);
    }
    if (mapOptimizer != nullptr)
        mapOptimizerThread_ = std::make_shared<std::thread>(&MapOptimizerBase::run, mapOptimizer);

    // Output welcome message
            std::cout << std::endl <<
                      "OpenGV2 Library Version " << PROJECT_VERSION_MAJOR << "." << PROJECT_VERSION_MINOR << std::endl;
}

void opengv2::SystemBase::shutdown() {
    if (viewer != nullptr) {
        viewer->requestFinish();
        viewerThread_->join();
    }
    if (mapOptimizer != nullptr) {
        mapOptimizer->requestFinish();
        mapOptimizerThread_->join();
    }
}

void opengv2::SystemBase::evaluation(int sensorId, std::vector<double> &result) {
    if (!GT_Tws_[sensorId].empty()) {
        std::vector<double> error_R, error_t;

        map->keyframeLockShared();
        for (auto it1 = map->keyframes().begin(), it2 = ++(map->keyframes().begin());
             it2 != map->keyframes().end(); it1 = it2++) {
            while (it2 != map->keyframes().end() && !(it2->second->isKeyFrame))
                it2++;
            if (it2 == map->keyframes().end())
                break;

            auto bf1 = it1->second;
            auto bf2 = it2->second;
            Eigen::Quaterniond unitQws1 = bf1->unitQwb() * Bodyframe::unitQsb(sensorId).conjugate();
            Eigen::Vector3d tws1 =
                    bf1->twb() - bf1->unitQwb() * (Bodyframe::unitQsb(sensorId).conjugate() * Bodyframe::tsb(sensorId));
            Eigen::Quaterniond unitQws2 = bf2->unitQwb() * Bodyframe::unitQsb(sensorId).conjugate();
            Eigen::Vector3d tws2 =
                    bf2->twb() - bf2->unitQwb() * (Bodyframe::unitQsb(sensorId).conjugate() * Bodyframe::tsb(sensorId));
            Eigen::Quaterniond unitQs1s2 = unitQws1.conjugate() * unitQws2;
            Eigen::Vector3d ts1s2 = unitQws1.conjugate() * (tws2 - tws1);
            ts1s2.normalize();

            // GT
            const Eigen::Matrix<double, 7, 1> &GT_Tws1 = GT_Tws_[sensorId].lower_bound(bf1->timeStamp())->second;
            const Eigen::Matrix<double, 7, 1> &GT_Tws2 = GT_Tws_[sensorId].lower_bound(bf2->timeStamp())->second;
            Eigen::Map<const Eigen::Quaterniond> GT_Qws1(GT_Tws1.data());
            Eigen::Map<const Eigen::Vector3d> GT_tws1(GT_Tws1.data() + 4);
            Eigen::Map<const Eigen::Quaterniond> GT_Qws2(GT_Tws2.data());
            Eigen::Map<const Eigen::Vector3d> GT_tws2(GT_Tws2.data() + 4);
            Eigen::Quaterniond GT_unitQs1s2 = GT_Qws1.conjugate() * GT_Qws2;
            Eigen::Vector3d GT_ts1s2 = GT_Qws1.conjugate() * (GT_tws2 - GT_tws1);
            GT_ts1s2.normalize();

            error_R.push_back(std::acos(std::min(1., std::max(-1., (((GT_unitQs1s2.conjugate() *
                                                                      unitQs1s2).matrix().trace() - 1.) / 2.)))) * 180 /
                              M_PI);
            error_t.push_back((GT_ts1s2 - ts1s2).norm());
        }
        map->keyframeUnlockShared();
        result.clear();
        {
            double sum = std::accumulate(error_t.begin(), error_t.end(), 0.0);
            double mean = sum / error_t.size();
            std::vector<double> diff(error_t.size());
            std::transform(error_t.begin(), error_t.end(), diff.begin(), [mean](double x) { return x - mean; });
            double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            double stdev = std::sqrt(sq_sum / (error_t.size() - 1));
            result.push_back(mean);
            result.push_back(stdev);
        }
        {
            double sum = std::accumulate(error_R.begin(), error_R.end(), 0.0);
            double mean = sum / error_R.size();
            std::vector<double> diff(error_R.size());
            std::transform(error_R.begin(), error_R.end(), diff.begin(), [mean](double x) { return x - mean; });
            double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            double stdev = std::sqrt(sq_sum / (error_R.size() - 1));
            result.push_back(mean);
            result.push_back(stdev);
        }
    }
}

void opengv2::SystemBase::saveKeyFrameTrajectoryTUM(int sensorId, const std::string &filename) {
    std::cout << std::endl << "Saving keyframe trajectory to " << filename << " ..." << std::endl;

    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    map->keyframeLockShared();
    int num = map->keyframes().size();
    for (auto &it: map->keyframes()) {
        Bodyframe::Ptr bf = it.second;
        if (!bf->isKeyFrame)
            continue;
        Eigen::Vector3d twb = bf->twb();
        Eigen::Quaterniond Qwb = bf->unitQwb();

        Eigen::Vector3d tbc = -(Bodyframe::unitQsb(sensorId).conjugate() * Bodyframe::tsb(sensorId));
        Eigen::Vector3d twc = Qwb * tbc + twb;
        Eigen::Quaterniond Qwc = Qwb * Bodyframe::unitQsb(sensorId).conjugate();

        //timestamp tx ty tz qx qy qz qw
        f << std::setprecision(10) << bf->timeStamp() << " " << twc[0] << " " << twc[1] << " "
          << twc[2] << " " << Qwc.x() << " " << Qwc.y() << " " << Qwc.z() << " " << Qwc.w() << std::endl;
    }
    map->keyframeUnlockShared();

    f.close();
    std::cout << std::endl << num << " frames trajectory saved!" << std::endl;
}

void opengv2::SystemBase::loadTUM(const std::string &file, std::map<double, Eigen::Matrix < double, 7, 1>>

&T) {
T.

clear();

//TUM: timestamp tx ty tz qx qy qz qw
std::ifstream fInput(file, std::ifstream::in);

std::vector<double> v(8);
char line[512];
fInput.
getline(line,
512);
while (!fInput.

eof() &&

!fInput.

fail()

) {
std::stringstream ss(line, std::stringstream::in);
for (
int i = 0;
i < 8; ++i) {
            ss >> v[i];
        }

        Eigen::Quaterniond Qwc(v.data() + 4);
Qwc.

normalize(); // normalize for further usage
Eigen::Matrix<double, 7, 1> vT;
vT.head(4) = Qwc.

coeffs();

vT.tail(3) << v[1], v[2], v[3];
T[v[0]] =
std::move(vT);

fInput.
getline(line,
512);
}
}

void opengv2::SystemBase::loadGT(int sensorId, std::map<double, Eigen::Matrix < double, 7, 1>>

&GT_Tws) {
GT_Tws_[sensorId] =
std::move(GT_Tws);
}

const std::map<double, Eigen::Matrix < double, 7, 1>> &

opengv2::SystemBase::GT_Tws(int sensorId) {
    return GT_Tws_[sensorId];
}

void opengv2::SystemBase::applyGTscale() {
    for (int i = 0; i < GT_Tws_.size(); i++) {
        if (!GT_Tws_[i].empty()) {
            std::vector<Eigen::Vector3d> tb1b2;
            map->keyframeLockShared();
            auto it1 = map->keyframes().begin();
            auto it2 = it1;
            for (it2++; it2 != map->keyframes().end(); it1++, it2++) {
                auto bf1 = it1->second;
                auto bf2 = it2->second;
                tb1b2.emplace_back(bf1->unitQwb().conjugate() * bf2->twb() - (bf1->unitQwb().conjugate() * bf1->twb()));
                tb1b2.back().normalize();

                // GT scale
                const Eigen::Matrix<double, 7, 1> &GT_Tws1 = GT_Tws_[i].lower_bound(bf1->timeStamp())->second;
                const Eigen::Matrix<double, 7, 1> &GT_Tws2 = GT_Tws_[i].lower_bound(bf2->timeStamp())->second;
                Eigen::Map<const Eigen::Quaterniond> GT_Qws1(GT_Tws1.data());
                Eigen::Map<const Eigen::Vector3d> GT_tws1(GT_Tws1.data() + 4);
                Eigen::Vector3d GT_twb1 = GT_Qws1 * Bodyframe::tsb(i) + GT_tws1;

                Eigen::Map<const Eigen::Quaterniond> GT_Qws2(GT_Tws2.data());
                Eigen::Map<const Eigen::Vector3d> GT_tws2(GT_Tws2.data() + 4);
                Eigen::Vector3d GT_twb2 = GT_Qws2 * Bodyframe::tsb(i) + GT_tws2;

                // rescale
                tb1b2.back() *= (GT_twb2 - GT_twb1).norm();
            }

            it1 = map->keyframes().begin();
            it2 = it1;
            it2++;
            for (int j = 0; it2 != map->keyframes().end(); it1++, it2++, j++) {
                auto bf1 = it1->second;
                auto bf2 = it2->second;

                bf2->setPose(bf1->unitQwb() * tb1b2[j] + bf1->twb(), bf2->unitQwb());
            }

            map->keyframeUnlockShared();
            break;
        }
    }
}

void opengv2::SystemBase::transformIdentity() {
    map->keyframeLockShared();
    auto bf0 = map->keyframes().begin()->second;
    Eigen::Quaterniond unitQ = bf0->unitQwb().conjugate();
    Eigen::Vector3d t = -(bf0->unitQwb().conjugate() * bf0->twb());
    for (auto &it : map->keyframes()) {
        auto bf = it.second;
        bf->setPose(unitQ * bf->twb() + t, unitQ * bf->unitQwb());
    }
    map->keyframeUnlockShared();
    map->landmarkLockShared();
    for (auto &it : map->landmarks()) {
        auto lm = it.second;
        lm->setPosition(unitQ * lm->position() + t);
    }
    map->landmarkUnlockShared();
}
