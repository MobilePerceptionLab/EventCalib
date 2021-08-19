//
// Created by huangkun on 2020/10/11.
//

#include <thread>
#include <ceres/ceres.h>

#include <opengv2/event_camera_calib/EventCalibSpline.hpp>
#include <opengv2/event_camera_calib/CirclesEventFrame.hpp>
#include <opengv2/sensor/PinholeCamera.hpp>
#include <opengv2/utility/utility.hpp>
#include <KDTreeVectorOfVectorsAdaptor.h>

opengv2::EventCalibSpline::EventCalibSpline(opengv2::MapBase::Ptr map, opengv2::EventContainer::Ptr eventContainer,
                                            bool useSO3, bool reduceMapb, double motionTimeStep, double circleRadius)
        : map_(map), eventContainer_(eventContainer), useSO3_(useSO3), motionTimeStep_(motionTimeStep),
          circleRadius_(circleRadius), reduceMap_(reduceMapb) {
    relationContainer_.clear();
    /** detect long-time gaps **/
    std::vector<vectorofEigenMatrix<Eigen::Vector3d>> twbSets;
    std::vector<vectorofEigenMatrix<Sophus::SO3d>> QwbSO3Sets;
    std::vector<vectorofEigenMatrix<Eigen::Matrix<double, 4, 1>>> QwbSets;
    std::vector<std::vector<double>> timeStampSets;

    if (map_->frameNum() <= 10) {
        throw std::logic_error("too few frames in the map.");
    }

    reduceMap();

    timeStampSets.reserve(sampleIdSets_.size());
    twbSets.reserve(sampleIdSets_.size());
    if (useSO3_)
        QwbSO3Sets.reserve(sampleIdSets_.size());
    else
        QwbSets.reserve(sampleIdSets_.size());
    for (const auto &sampleSet : sampleIdSets_) {
        timeStampSets.emplace_back();
        timeStampSets.back().reserve(sampleSet.size());
        twbSets.emplace_back();
        twbSets.back().reserve(sampleSet.size());
        if (useSO3_) {
            QwbSO3Sets.emplace_back();
            QwbSO3Sets.back().reserve(sampleSet.size());
        } else {
            QwbSets.emplace_back();
            QwbSets.back().reserve(sampleSet.size());
        }

        for (const auto &id: sampleSet) {
            auto bf = map_->keyframe(id);
            twbSets.back().push_back(bf->twb());
            if (useSO3_)
                QwbSO3Sets.back().emplace_back(bf->unitQwb());
            else
                QwbSets.back().emplace_back(bf->unitQwb().coeffs()); // x y z w
            timeStampSets.back().push_back(bf->timeStamp());
        }
    }

    /** initialize splines **/
    // extend time bounds, since spline can only evaluate inside the bound.
    for (auto &timeStampSet: timeStampSets) {
        timeStampSet.front() -= 3 * motionTimeStep_;
        timeStampSet.back() += 3 * motionTimeStep_;
    }

    for (int i = 0; i < timeStampSets.size(); ++i) {
        time2splineIdx_.emplace_back(timeStampSets[i].front(), timeStampSets[i].back());
    }

    twbSplines_.reserve(twbSets.size());
    if (useSO3_)
        QwbSO3Splines_.reserve(QwbSO3Sets.size());
    else
        QwbSplines_.reserve(QwbSets.size());
    for (int i = 0; i < timeStampSets.size(); ++i) {
        if (timeStampSets[i].size() < degree_ + 1) {
            throw std::logic_error("sampleSets not filtered");
        }
        int cpNum = std::floor((timeStampSets[i].back() - timeStampSets[i].front()) / (50 * motionTimeStep_));
        if (cpNum > timeStampSets[i].size())
            cpNum = timeStampSets[i].size() - 1;
        if (cpNum < degree_ + 1)
            cpNum = degree_ + 1; // become a bezier curve
        twbSplines_.emplace_back(3, twbSets[i], cpNum, timeStampSets[i]);
        if (useSO3_)
            QwbSO3Splines_.emplace_back(degree_, QwbSO3Sets[i], cpNum, timeStampSets[i]);
        else
            QwbSplines_.emplace_back(degree_, QwbSets[i], cpNum, timeStampSets[i]);
    }

    // setup intrinsics_
    auto camera = dynamic_cast<PinholeCamera *>(eventContainer_->camera.get());
    const Eigen::Matrix3d &K = camera->K();
    const Eigen::VectorXd &distCoeffs = camera->distCoeffs();
    if (distCoeffs.size() != 5) {
        throw std::logic_error("Sorry! Only radial distortion considered for now.");
    }

    Eigen::Vector4d radialDistortion(distCoeffs(0), distCoeffs(1), distCoeffs(4), 0);
    Eigen::VectorXd inverseRadial = PinholeCamera::inverseRadialDistortion(radialDistortion);

    intrinsics_ << K(0, 0), K(1, 1), K(0, 2), K(1, 2),
            inverseRadial(0), inverseRadial(1), inverseRadial(2), inverseRadial(3), inverseRadial(4);

    std::cout << "OpenCV Distortion before optimization:" << radialDistortion.transpose() << std::endl;
    std::cout << "Intrinsics before optimization:" << intrinsics_.transpose() << std::endl;

    optimize();

    updateMap();
}

bool opengv2::EventCalibSpline::optimize() {
    ceres::Problem problem;
    ceres::LocalParameterization *SO3_parameterization = new LocalParameterizationSO3();
    ceres::LocalParameterization *quaternion_parameterization = new ceres::EigenQuaternionParameterization();

    // Specify local update rule for our parameter
    if (useSO3_) {
        for (auto &QwbSpline: QwbSO3Splines_) {
            for (Sophus::SO3d &cp:QwbSpline.controlPoints()) {
                problem.AddParameterBlock(cp.data(), Sophus::SO3d::num_parameters, SO3_parameterization);
            }
        }
        delete quaternion_parameterization;
    } else {
        for (auto &QwbSpline: QwbSplines_) {
            for (Eigen::Matrix<double, 4, 1> &cp:QwbSpline.getCP()) {
                problem.AddParameterBlock(cp.data(), 4, quaternion_parameterization);
            }
        }
        delete SO3_parameterization;
    }

    // TODO: merge similar event in time use the the same basis.

    // give each event a reference frame w.r.t timeStamp
    std::vector<Eigen::Matrix<double, 1, 1>> timeStamps;
    std::unordered_map<size_t, double> idx2FrameId;
    timeStamps.reserve(map_->frameNum());
    idx2FrameId.reserve(map_->frameNum());
    size_t counter = 0;
    map_->keyframeLockShared();
    for (const auto &pair: map_->keyframes()) {
        timeStamps.emplace_back(pair.second->timeStamp());
        idx2FrameId.emplace(counter, pair.first);

        counter++;
    }
    map_->keyframeUnlockShared();
    KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Matrix<double, 1, 1>>, double, 1, nanoflann::metric_L2_Simple>
            frameSearchingTree(1, timeStamps);

    // pre-calculate spline basis
    relationContainer_.reserve(eventContainer_->container.size() / 3);
    for (int i = 0; i < twbSplines_.size(); ++i) {
        for (auto itr = eventContainer_->container.lower_bound(twbSplines_[i].getCorrespondingUs().front());
             itr != eventContainer_->container.upper_bound(twbSplines_[i].getCorrespondingUs().back()); /**/ itr++) {
            double u = itr->first;

            // decide event belong to which frame
            size_t idx;
            double sqrDistance;
            frameSearchingTree.query(&u, 1, &idx, &sqrDistance);

            if (sqrDistance < (5 * motionTimeStep_ * 5 * motionTimeStep_)) {
                auto cf = dynamic_cast<CirclesEventFrame *>(map_->keyframe(idx2FrameId[idx])->frame(0).get());
                if (auto lm = cf->findCenter(itr->second.location)) {
                    auto rbasisFun = std::make_shared<std::vector<std::vector<double>>>();
                    auto tbasisFun = std::make_shared<std::vector<std::vector<double>>>();
                    size_t rSpanIdx = useSO3_ ? QwbSO3Splines_[i].findSpan(u) : QwbSplines_[i].findSpan(u);
                    size_t tSpanIdx = twbSplines_[i].findSpan(u);
                    if (useSO3_)
                        QwbSO3Splines_[i].derBasisFuns(u, rSpanIdx, 0, *rbasisFun);
                    else
                        QwbSplines_[i].dersBasisFuns(u, rSpanIdx, 0, *rbasisFun);
                    twbSplines_[i].dersBasisFuns(u, tSpanIdx, 0, *tbasisFun);

                    relationContainer_.emplace_back(itr->second.location, lm->position(),
                                                    rbasisFun, tbasisFun,
                                                    rSpanIdx, tSpanIdx, i);
                    // itr++;
                } else {
                    // itr = eventContainer_->container.erase(itr);
                }
            } else {
                // itr = eventContainer_->container.erase(itr);
            }
        }
    }

    eventContainer_->container.clear();

    // Create and add cost functions. Derivatives will be evaluated via automatic differentiation
    ceres::LossFunction *huber = new ceres::HuberLoss(circleRadius_ * 0.2);
    Eigen::Quaterniond Qbs = Bodyframe::unitQsb(0).conjugate();
    Eigen::Vector3d tbs = -(Qbs * Bodyframe::tsb(0));
    for (const auto &relation: relationContainer_) {
        const auto &splineIdx = relation.splineIdx;
        const auto &rSpanIdx = relation.rSpanIdx;
        const auto &tSpanIdx = relation.tSpanIdx;
        if (useSO3_) {
            ceres::CostFunction *cost_function = CalibReprojectionError_SO3::Create(
                    &relation.obs,
                    &relation.lm, &circleRadius_,
                    &Qbs, &tbs,
                    relation.rBasis, relation.tBasis);
            problem.AddResidualBlock(cost_function, huber, intrinsics_.data(),
                                     QwbSO3Splines_[splineIdx].controlPoints()[rSpanIdx - 3 + 0].data(),
                                     QwbSO3Splines_[splineIdx].controlPoints()[rSpanIdx - 3 + 1].data(),
                                     QwbSO3Splines_[splineIdx].controlPoints()[rSpanIdx - 3 + 2].data(),
                                     QwbSO3Splines_[splineIdx].controlPoints()[rSpanIdx - 3 + 3].data(),
                                     twbSplines_[splineIdx].getCP()[tSpanIdx - 3 + 0].data(),
                                     twbSplines_[splineIdx].getCP()[tSpanIdx - 3 + 1].data(),
                                     twbSplines_[splineIdx].getCP()[tSpanIdx - 3 + 2].data(),
                                     twbSplines_[splineIdx].getCP()[tSpanIdx - 3 + 3].data());
        } else {
            ceres::CostFunction *cost_function = CalibReprojectionError::Create(
                    &relation.obs,
                    &relation.lm, &circleRadius_,
                    &Qbs, &tbs,
                    relation.rBasis, relation.tBasis);
            problem.AddResidualBlock(cost_function, huber, intrinsics_.data(),
                                     QwbSplines_[splineIdx].getCP()[rSpanIdx - 3 + 0].data(),
                                     QwbSplines_[splineIdx].getCP()[rSpanIdx - 3 + 1].data(),
                                     QwbSplines_[splineIdx].getCP()[rSpanIdx - 3 + 2].data(),
                                     QwbSplines_[splineIdx].getCP()[rSpanIdx - 3 + 3].data(),
                                     twbSplines_[splineIdx].getCP()[tSpanIdx - 3 + 0].data(),
                                     twbSplines_[splineIdx].getCP()[tSpanIdx - 3 + 1].data(),
                                     twbSplines_[splineIdx].getCP()[tSpanIdx - 3 + 2].data(),
                                     twbSplines_[splineIdx].getCP()[tSpanIdx - 3 + 3].data());
        }
    }

    // Set solver options (precision / method)
    ceres::Solver::Options options;
    options.gradient_tolerance = Sophus::Constants<double>::epsilon();
    options.function_tolerance = Sophus::Constants<double>::epsilon();
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = std::thread::hardware_concurrency() - 2;
    options.num_linear_solver_threads = std::thread::hardware_concurrency() - 2;

    // Solve
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    return true;
}

void opengv2::EventCalibSpline::updateMap() {
    auto camera = dynamic_cast<PinholeCamera *>(eventContainer_->camera.get());

    // Update intrinsics
    Eigen::Matrix3d K_new;
    K_new << intrinsics_(0), 0, intrinsics_(2),
            0, intrinsics_(1), intrinsics_(3),
            0, 0, 1;
    camera->setK(K_new);
    camera->inverseRadialPoly() = Eigen::Matrix<double, 5, 1>(intrinsics_.data() + 4);
    std::cout.precision(12);
    std::cout << "Intrinsics after optimization:" << intrinsics_.transpose() << std::endl;

    // Update Map
    map_->keyframeLockShared();
    for (const auto &pair: map_->keyframes()) {
        Bodyframe::Ptr bf = pair.second;
        int idx = time2splineIdx(bf->timeStamp());
        if (idx < 0)
            continue;

        vectorofEigenMatrix<Eigen::Vector3d> Ders;
        twbSplines_[idx].evaluate(bf->timeStamp(), 0, Ders);
        Eigen::Vector3d twb(Ders[0]);

        Eigen::Quaterniond Qbw;
        if (useSO3_) {
            Sophus::SO3d Rwb;
            std::vector<Eigen::Vector3d> unused;
            QwbSO3Splines_[idx].evaluate(bf->timeStamp(), 0, Rwb, unused);
            Qbw = Rwb.unit_quaternion().conjugate();
        } else {
            vectorofEigenMatrix<Eigen::Vector4d> ders;
            QwbSplines_[idx].evaluate(bf->timeStamp(), 0, ders);
            ders[0].normalize();
            Eigen::Map<Eigen::Quaterniond> q(ders[0].data());
            Qbw = q.conjugate();
        }

        bf->optT.clear();
        std::copy(Qbw.coeffs().data(), Qbw.coeffs().data() + 4, std::back_inserter(bf->optT));
        std::copy(twb.data(), twb.data() + 3, std::back_inserter(bf->optT));
    }
    // switch
    for (const auto &it: map_->keyframes()) {
        Bodyframe::Ptr bf = it.second;

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
    map_->keyframeUnlockShared();
}

void opengv2::EventCalibSpline::reduceMap() {
    sampleIdSets_.clear();
    sampleIdSets_.emplace_back();
    map_->keyframeLockShared();
    double lastTimeStamp = map_->keyframes().begin()->second->timeStamp();
    for (auto &itr : map_->keyframes()) {
        auto bf = itr.second;
        if (bf->timeStamp() - lastTimeStamp > 50 * motionTimeStep_) {
            sampleIdSets_.emplace_back();
        }

        sampleIdSets_.back().push_back(bf->timeStamp());
        lastTimeStamp = bf->timeStamp();
    }
    map_->keyframeUnlockShared();

    // segment samples less than degree_+1 will be discarded.
    for (auto itr = sampleIdSets_.begin(); itr != sampleIdSets_.end();) {
        if (itr->size() < degree_ + 1) {
            for (double id: *itr) {
                map_->removeFrame(id);
            }
            itr = sampleIdSets_.erase(itr);
        } else {
            itr++;
        }
    }

    if (!reduceMap_)
        return;
    //TODO: The following content currently disabled, since not works well.

    // building KD-tree for R and t
    vectorofEigenMatrix<Eigen::Vector3d> twbSet;
    vectorofEigenMatrix<Eigen::Quaterniond> QwbSet;
    std::unordered_map<int, std::pair<int, int>> idx2splineIdxAndSubIdx;
    std::unordered_map<double, bool> erased; // flag record status
    erased.reserve(map_->frameNum());
    for (int splineIdx = 0; splineIdx < sampleIdSets_.size(); ++splineIdx) {
        for (int subIdx = 0; subIdx < sampleIdSets_[splineIdx].size(); ++subIdx) {
            auto bf = map_->keyframe(sampleIdSets_[splineIdx][subIdx]);
            twbSet.push_back(bf->twb());
            QwbSet.push_back(bf->unitQwb());
            idx2splineIdxAndSubIdx[twbSet.size() - 1] = std::pair(splineIdx, subIdx);

            erased.emplace(bf->timeStamp(), false);
        }
    }
    KDTreeVectorOfVectorsAdaptor<vectorofEigenMatrix<Eigen::Vector3d>, double, 3, nanoflann::metric_L2_Simple>
            t_kdTree(3, twbSet, 10);
    nanoflann::SO3DataSetAdaptor QwbSetAdapter(QwbSet);
    nanoflann::SO3_KDTree<double> R_kdTree(4, QwbSetAdapter, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    R_kdTree.buildIndex();

    double durationThreshold = 20 * motionTimeStep_; // piece duration should larger than this.
    double R_similarThreshold = 0.005 * 0.005; // squared L2 unit quaternion distance // TODO: to be check
    double t_similarThreshold = (circleRadius_) * (circleRadius_); // squared L2 distance

    std::unordered_map<double, std::vector<std::pair<int, std::pair<int, int>>>> similarityMap;
    std::vector<std::pair<size_t, double>> t_IndicesDists, R_IndicesDists;
    for (int splineIdx = 0; splineIdx < sampleIdSets_.size(); ++splineIdx) {
        for (int subIdx = 0; subIdx < sampleIdSets_[splineIdx].size(); ++subIdx) {
            if (erased[sampleIdSets_[splineIdx][subIdx]])
                continue;
            auto bf = map_->keyframe(sampleIdSets_[splineIdx][subIdx]);
            similarityMap.emplace(bf->timeStamp(), std::vector<std::pair<int, std::pair<int, int>>>());

            // kd-tree searching
            t_kdTree.index->radiusSearch(bf->twb().data(), t_similarThreshold, t_IndicesDists,
                                         nanoflann::SearchParams(32, 0, false));
            R_kdTree.radiusSearch(bf->unitQwb().coeffs().data(), R_similarThreshold, R_IndicesDists,
                                  nanoflann::SearchParams(32, 0, false));

            // merge two searching result
            std::vector<std::set<int>> idxSets(sampleIdSets_.size()); // assuming idxSet ascending order
            std::unordered_set<size_t> tIdxs;
            for (const auto &pair: t_IndicesDists) {
                tIdxs.insert(pair.first);
            }
            for (const auto &pair: R_IndicesDists) {
                if (tIdxs.find(pair.first) != tIdxs.end()) {
                    const auto &p = idx2splineIdxAndSubIdx[pair.first];
                    if (!erased[sampleIdSets_[p.first][p.second]])
                        idxSets[p.first].insert(p.second);
                }
            }

            for (int spline_idx = 0; spline_idx < idxSets.size(); spline_idx++) {
                const auto &idxSet = idxSets[spline_idx];
                if (idxSet.empty())
                    continue;

                std::vector<std::pair<int, int>> bounds;
                bounds.emplace_back(*idxSet.begin(), *idxSet.begin());
                for (double idx : idxSet) {
                    if (idx - bounds.back().second > 2) {
                        bounds.emplace_back(idx, idx);
                    }
                    bounds.back().second = idx;
                }
                for (const auto &pair: bounds) {
                    if (map_->keyframe(sampleIdSets_[spline_idx][pair.second])->timeStamp() -
                        map_->keyframe(sampleIdSets_[spline_idx][pair.first])->timeStamp() > durationThreshold) {
                        // segment contain constant parts: delete such pieces.
                        for (int idx = pair.first; idx <= pair.second; ++idx) {
                            erased[sampleIdSets_[spline_idx][idx]] = true;
                        }
                    } else {
                        // store for future process
                        similarityMap[bf->timeStamp()].emplace_back(spline_idx, pair);
                    }
                }
            }
        }
    }

    std::vector<std::vector<bool>> valued(sampleIdSets_.size());
    for (int splineIdx = 0; splineIdx < sampleIdSets_.size(); ++splineIdx) {
        valued[splineIdx].assign(sampleIdSets_[splineIdx].size(), false);
    }
    for (int splineIdx = 0; splineIdx < sampleIdSets_.size(); ++splineIdx) {
        for (int subIdx = 0; subIdx < sampleIdSets_[splineIdx].size(); ++subIdx) {
            if (erased[sampleIdSets_[splineIdx][subIdx]])
                continue;
            auto bf = map_->keyframe(sampleIdSets_[splineIdx][subIdx]);
            std::vector<std::vector<bool>> valuedSets(valued);

            valuedSets[splineIdx][subIdx] = true;
            for (const auto &tri: similarityMap[bf->timeStamp()]) { // start from a single sample
                const auto &spline_idx = tri.first;
                const auto &bound = tri.second;
                for (int idx = bound.first; idx <= bound.second; ++idx) {
                    valuedSets[spline_idx][idx] = true;
                    auto id = sampleIdSets_[spline_idx][idx];
                    if (!erased[id]) {
                        for (const auto &triple: similarityMap[id]) {
                            const auto &n_spline_idx = triple.first;
                            const auto &n_bound = triple.second;
                            for (int idx1 = n_bound.first; idx1 <= n_bound.second; ++idx1) {
                                valuedSets[n_spline_idx][idx1] = true;
                            }
                        }
                    }
                }
            }

            // merge nearby bounds
            std::vector<std::vector<std::pair<int, int>>> boundSets(valuedSets.size());
            for (int m_spline_idx = 0; m_spline_idx < valuedSets.size(); m_spline_idx++) {
                auto &bounds = boundSets[m_spline_idx];
                const auto &valuedSet = valuedSets[m_spline_idx];
                for (int idx = 0; idx < valuedSet.size(); ++idx) {
                    if (valuedSet[idx]) {
                        if (bounds.empty()) {
                            bounds.emplace_back(idx, idx);
                        } else {
                            if (idx - bounds.back().second > 2) {
                                bounds.emplace_back(idx, idx);
                            }
                            bounds.back().second = idx;
                        }
                    }
                }
            }

            // segments have similar parts: keep longest, delete others.
            std::pair<int, int> longestBoundIdx(0, 0);
            double longestBoundDuration = 0;
            for (int m_spline_idx = 0; m_spline_idx < boundSets.size(); m_spline_idx++) {
                const auto &bounds = boundSets[m_spline_idx];
                for (int boundIdx = 0; boundIdx < bounds.size(); ++boundIdx) {
                    const auto &pair = bounds[boundIdx];
                    double duration = map_->keyframe(sampleIdSets_[m_spline_idx][pair.second])->timeStamp() -
                                      map_->keyframe(sampleIdSets_[m_spline_idx][pair.first])->timeStamp();
                    if (duration > durationThreshold) {
                        for (int idx2 = pair.first; idx2 <= pair.second; ++idx2) {
                            erased[sampleIdSets_[m_spline_idx][idx2]] = true;
                        }

                        if (duration > longestBoundDuration) {
                            longestBoundIdx = std::pair(m_spline_idx, boundIdx);
                            longestBoundDuration = duration;
                        }
                    }
                }
            }
            if (longestBoundDuration > durationThreshold) {
                const auto &pair = boundSets[longestBoundIdx.first][longestBoundIdx.second];
                for (int idx2 = pair.first; idx2 <= pair.second; ++idx2) {
                    erased[sampleIdSets_[longestBoundIdx.first][idx2]] = false;
                }
            }
        }
    }

    // update sampleIdSets_ by erased[]
    bool generateNewSet = true;
    std::vector<std::vector<double>> newSampleIdSets;
    for (auto &sampleIdSet : sampleIdSets_) {
        for (double &sampleId : sampleIdSet) {
            if (erased[sampleId]) {
                generateNewSet = true;
                continue;
            } else {
                if (generateNewSet) {
                    generateNewSet = false;
                    newSampleIdSets.emplace_back();
                }
                newSampleIdSets.back().push_back(sampleId);
            }
        }
    }
    sampleIdSets_ = std::move(newSampleIdSets);

    // segment samples less than degree_+1 will be discarded.
    for (auto itr = sampleIdSets_.begin(); itr != sampleIdSets_.end();) {
        if (itr->size() < degree_ + 1) {
            for (auto id: *itr) {
                map_->removeFrame(id);
            }
            itr = sampleIdSets_.erase(itr);
        } else {
            itr++;
        }
    }

    // update map
    for (const auto &pair: erased) {
        if (pair.second) {
            map_->removeFrame(pair.first);
        }
    }
}