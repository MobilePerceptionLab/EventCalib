//
// Created by huangkun on 2020/9/18.
//
#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <algorithm>
#include <limits>

#include <cv_calib.hpp>
#include <dbscan.h>
#include <opengv2/utility/utility.hpp>
#include <opengv2/event_camera_calib/CirclesEventFrame.hpp>
#include <opengv2/event_camera_calib/CalibCircle.hpp>
#include <opengv2/sensor/PinholeCamera.hpp>

opengv2::CirclesEventFrame::CirclesEventFrame(EventContainer::Ptr container, const std::pair<double, double> &duration,
                                              CirclePatternParameters::Ptr pattern, Params params) :
        EventFrame(container, duration), pattern_(pattern), params_(params) {
    auto camera = dynamic_cast<CameraBase *>(sensor_.get());
    if (pattern_->isAsymmetric) {
        circleRadiusThreshold_ = std::min(
                std::max(camera->size()[0], camera->size()[1]) /
                std::max(pattern_->rows, 2 * pattern_->cols),
                std::min(camera->size()[0], camera->size()[1]) /
                std::min(pattern_->rows, 2 * pattern_->cols)) / pattern->squareSize * pattern->circleRadius * 1.5;
    } else {
        circleRadiusThreshold_ = std::min(
                std::max(camera->size()[0], camera->size()[1]) /
                std::max(pattern_->rows, pattern_->cols),
                std::min(camera->size()[0], camera->size()[1]) /
                std::min(pattern_->rows, pattern_->cols)) / pattern->squareSize * pattern->circleRadius * 1.5;
    }
}

opengv2::CirclesEventFrame::Params::Params() {
    dbscan_eps = 4;
    dbscan_startMinSample = 2;
    clusterMinSample = 5;
    knn_num = 3;
}

opengv2::CirclesEventFrame::Params::Params(const cv::FileStorage &node) {
    node["dbscan_eps"] >> dbscan_eps;
    node["dbscan_startMinSample"] >> dbscan_startMinSample;
    node["clusterMinSample"] >> clusterMinSample;
    node["knn_num"] >> knn_num;
    node["fitCircle"] >> fitCircle;
}

struct CircleErr {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CircleErr(double fitErr, double radius, const Eigen::Ref<const Eigen::Vector2d> &center) :
            fitErr(fitErr), radius(radius), center(center) {}

    double fitErr;
    double radius;
    Eigen::Vector2d center;
};

bool opengv2::CirclesEventFrame::extractFeatures() {
    if (positiveEvents_.empty() || negativeEvents_.empty()) {
        return false;
    }

    auto dbscan = DBSCAN<Eigen::Vector2d, double>();
    dbscan.Run(&positiveEvents_, 2, params_.dbscan_eps, params_.dbscan_startMinSample);
    auto p_noise = std::move(dbscan.Noise);
    auto p_clusters = std::move(dbscan.Clusters);
    dbscan.Run(&negativeEvents_, 2, params_.dbscan_eps, params_.dbscan_startMinSample);
    auto n_noise = std::move(dbscan.Noise);
    auto n_clusters = std::move(dbscan.Clusters);

    /** Draw the result and filter data **/
    auto camera = dynamic_cast<CameraBase *>(sensor_.get());
    image_ = cv::Mat(camera->size()[1], camera->size()[0], CV_8UC3, cv::Vec3b(0, 0, 0));

    //debug
    eventImage = image_.clone();
    for (const Eigen::Vector2d &p: positiveEvents_) {
        cv::Point loc(p[0], p[1]);
        eventImage.at<cv::Vec3b>(loc) = cv::Vec3b(0, 0, 255);
    }
    for (const Eigen::Vector2d &p: negativeEvents_) {
        cv::Point loc(p[0], p[1]);
        eventImage.at<cv::Vec3b>(loc) = cv::Vec3b(0, 255, 0);
    }

    for (auto itr = p_clusters.begin(); itr != p_clusters.end();) {
        // remove cluster with too few samples
        if (itr->size() < params_.clusterMinSample) {
            itr = p_clusters.erase(itr);
        } else {
            unsigned int color = 20 * (itr - p_clusters.begin());
            for (unsigned int idx : *itr) {
                cv::Point loc(positiveEvents_[idx][0], positiveEvents_[idx][1]);
                image_.at<cv::Vec3b>(loc) = cv::Vec3b(color / 256, color % 256, 200);
            }

            itr++;
        }
    }

    for (auto itr = n_clusters.begin(); itr != n_clusters.end();) {
        // remove cluster with too few samples
        if (itr->size() < params_.clusterMinSample) {
            itr = n_clusters.erase(itr);
        } else {
            unsigned int color = 20 * (itr - n_clusters.begin());
            for (unsigned int idx : *itr) {
                cv::Point loc(negativeEvents_[idx][0], negativeEvents_[idx][1]);
                image_.at<cv::Vec3b>(loc) = cv::Vec3b(color / 256, color % 256, 100);
            }

            itr++;
        }
    }

    // store
    nClusters_ = n_clusters;
    pClusters_ = p_clusters;

    // debug
    clusterImage = image_.clone();

    // too few events
    if (p_clusters.size() < pattern_->rows * pattern_->cols || n_clusters.size() < pattern_->rows * pattern_->cols) {
        return false;
    }

    /** find candidate circles **/
    std::vector<std::pair<size_t, size_t>> candidates;
    vectorofEigenMatrix<Eigen::Vector2d> candidateCenters;
    std::vector<double> candidatesRadius;

    // calculate each cluster center (median)
    std::vector<uint> p_centers, n_centers;
    auto p_compare_fun = [&](uint lhs, uint rhs) { return positiveEvents_[lhs].norm() < positiveEvents_[rhs].norm(); };
    auto n_compare_fun = [&](uint lhs, uint rhs) { return negativeEvents_[lhs].norm() < negativeEvents_[rhs].norm(); };
    for (auto &pCluster: p_clusters) {
        std::nth_element(pCluster.begin(), pCluster.begin() + pCluster.size() / 2, pCluster.end(), p_compare_fun);
        p_centers.push_back(pCluster[pCluster.size() / 2]);
    }
    for (auto &nCluster: n_clusters) {
        std::nth_element(nCluster.begin(), nCluster.begin() + nCluster.size() / 2, nCluster.end(), n_compare_fun);
        n_centers.push_back(nCluster[nCluster.size() / 2]);
    }

    /*// draw cluster centers
    for (auto center: p_centers) {
        cv::Point loc(positiveEvents_[center][0], positiveEvents_[center][1]);
        cv::circle(image_, loc, 2, cv::Vec3b(255, 255, 255));
    }
    for (auto center: n_centers) {
        cv::Point loc(negativeEvents_[center][0], negativeEvents_[center][1]);
        cv::circle(image_, loc, 2, cv::Vec3b(255, 255, 255));
    }*/

    // building k-d tree
    vectorofEigenMatrix<Eigen::Vector2d> pCenters(p_centers.size()), nCenters(n_centers.size());
    for (int i = 0; i < p_centers.size(); ++i) {
        pCenters[i] = positiveEvents_[p_centers[i]];
    }
    for (int i = 0; i < n_centers.size(); ++i) {
        nCenters[i] = negativeEvents_[n_centers[i]];
    }
    KDTreeVectorOfVectorsAdaptor<vectorofEigenMatrix<Eigen::Vector2d>, double, 2, nanoflann::metric_L2_Simple>
            p_kdTree(2, pCenters, 10), n_kdTree(2, nCenters, 10);

    // Store fitError for Reusing
    std::unordered_map<std::pair<size_t, size_t>, CircleErr, pair_hash> fitErrandRadiusMap;

    const size_t num_results = params_.fitCircle ? params_.knn_num : 1;
    std::vector<size_t> n_idx(num_results);
    std::vector<size_t> p_idx(num_results);
    std::vector<double> out_dists_sqr(num_results);
    vectorofEigenMatrix<Eigen::Vector2d> centers(num_results);
    std::vector<double> radius(num_results);
    std::vector<double> fitErrs;
    if (params_.fitCircle) {
        for (size_t pi = 0; pi < p_centers.size(); ++pi) {
            size_t realNum_results = num_results;
            // search knn in opposite polarity cluster
            n_kdTree.query(positiveEvents_[p_centers[pi]].data(), num_results, n_idx.data(), out_dists_sqr.data());

            // remove cluster too far away
            for (int oi = 0; oi < out_dists_sqr.size(); ++oi) {
                if (out_dists_sqr[oi] > out_dists_sqr[0] * 4 ||
                    out_dists_sqr[oi] > 4 * circleRadiusThreshold_ * circleRadiusThreshold_) {
                    realNum_results = oi;
                    break;
                }
            }
            if (realNum_results == 0)
                continue;

            // connect two centers as diameter, verify circle fitting error
            fitErrs.assign(realNum_results, 0);
            for (int j = 0; j < realNum_results; ++j) {
                auto itr = fitErrandRadiusMap.find(std::pair<size_t, size_t>(pi, n_idx[j]));
                if (itr == fitErrandRadiusMap.end()) {
                    fitCircle(p_clusters[pi], n_clusters[n_idx[j]], centers[j], radius[j]);

                    double approxRadius =
                            (positiveEvents_[p_centers[pi]] - negativeEvents_[n_centers[n_idx[j]]]).norm() / 2;
                    if (radius[j] > circleRadiusThreshold_ || radius[j] > 2 * approxRadius) {
                        fitErrs[j] = std::numeric_limits<double>::max();
                    } else {
                        for (auto pEvent: p_clusters[pi]) {
                            fitErrs[j] += std::abs((positiveEvents_[pEvent] - centers[j]).norm() - radius[j]);
                        }
                        for (auto nEvent: n_clusters[n_idx[j]]) {
                            fitErrs[j] += std::abs((negativeEvents_[nEvent] - centers[j]).norm() - radius[j]);
                        }

                        fitErrs[j] /= (p_clusters[pi].size() + n_clusters[n_idx[j]].size()) * radius[j];
                    }
                    fitErrandRadiusMap.emplace(std::pair<size_t, size_t>(pi, n_idx[j]),
                                               CircleErr(fitErrs[j], radius[j], centers[j]));
                } else {
                    fitErrs[j] = itr->second.fitErr;
                    radius[j] = itr->second.radius;
                    centers[j] = itr->second.center;
                }
            }
            int n_minIdx = std::min_element(fitErrs.begin(), fitErrs.end()) - fitErrs.begin();

            // Do circle check for the minimum, if the fitting error greater than Expectation(assume gaussian noise)
            const double gaussianNoise = 2 / radius[n_minIdx]; // TODO: considering ellipse
            if (fitErrs[n_minIdx] < gaussianNoise) {
                /* Double Direction Check */
                realNum_results = num_results;
                p_kdTree.query(negativeEvents_[n_centers[n_idx[n_minIdx]]].data(), num_results, p_idx.data(),
                               out_dists_sqr.data());
                for (int oi = 0; oi < out_dists_sqr.size(); ++oi) {
                    if (out_dists_sqr[oi] > out_dists_sqr[0] * 4 ||
                        out_dists_sqr[oi] > 4 * circleRadiusThreshold_ * circleRadiusThreshold_) {
                        realNum_results = oi;
                        break;
                    }
                }
                if (realNum_results == 0)
                    continue;
                fitErrs.assign(realNum_results, 0);
                for (int i = 0; i < realNum_results; ++i) {
                    auto itr = fitErrandRadiusMap.find(std::pair<size_t, size_t>(p_idx[i], n_idx[n_minIdx]));
                    if (itr == fitErrandRadiusMap.end()) {
                        fitCircle(p_clusters[p_idx[i]], n_clusters[n_idx[n_minIdx]], centers[i], radius[i]);

                        double approxRadius = (positiveEvents_[p_centers[p_idx[i]]] -
                                               negativeEvents_[n_centers[n_idx[n_minIdx]]]).norm() / 2;
                        if (radius[i] > circleRadiusThreshold_ || radius[i] > 2 * approxRadius) {
                            fitErrs[i] = std::numeric_limits<double>::max();
                        } else {
                            for (auto pEvent: p_clusters[p_idx[i]]) {
                                fitErrs[i] += std::abs((positiveEvents_[pEvent] - centers[i]).norm() - radius[i]);
                            }
                            for (auto nEvent: n_clusters[n_idx[n_minIdx]]) {
                                fitErrs[i] += std::abs((negativeEvents_[nEvent] - centers[i]).norm() - radius[i]);
                            }
                            fitErrs[i] /=
                                    (p_clusters[p_idx[i]].size() + n_clusters[n_idx[n_minIdx]].size()) * radius[i];
                        }
                        fitErrandRadiusMap.emplace(std::pair<size_t, size_t>(p_idx[i], n_idx[n_minIdx]),
                                                   CircleErr(fitErrs[i], radius[i], centers[i]));
                    } else {
                        fitErrs[i] = itr->second.fitErr;
                        radius[i] = itr->second.radius;
                        centers[i] = itr->second.center;
                    }
                }
                int p_minIdx = std::min_element(fitErrs.begin(), fitErrs.end()) - fitErrs.begin();

                // if the same, add to candidate
                if (p_idx[p_minIdx] == pi) {
                    candidates.emplace_back(pi, n_idx[n_minIdx]);
                    candidateCenters.push_back(centers[p_minIdx]);
                    candidatesRadius.push_back(radius[p_minIdx]);
                }
            }
        }
    } else {
        for (size_t pi = 0; pi < p_centers.size(); ++pi) {
            n_kdTree.query(positiveEvents_[p_centers[pi]].data(), num_results, n_idx.data(), out_dists_sqr.data());
            // remove cluster too far away
            if (out_dists_sqr[0] > 4 * circleRadiusThreshold_ * circleRadiusThreshold_) {
                continue;
            }
            p_kdTree.query(negativeEvents_[n_centers[n_idx[0]]].data(), num_results, p_idx.data(),
                           out_dists_sqr.data());
            if (p_idx[0] == pi) {
                // check circle
                Eigen::Vector2d center =
                        (positiveEvents_[p_centers[p_idx[0]]] + negativeEvents_[n_centers[n_idx[0]]]) / 2;
                double r = (positiveEvents_[p_centers[p_idx[0]]] - negativeEvents_[n_centers[n_idx[0]]]).norm() / 2;
                double fitErr = 0;
                for (auto pEvent: p_clusters[p_idx[0]]) {
                    fitErr += std::abs((positiveEvents_[pEvent] - center).norm() - r);
                }
                for (auto nEvent: n_clusters[n_idx[0]]) {
                    fitErr += std::abs((negativeEvents_[nEvent] - center).norm() - r);
                }
                fitErr /= (p_clusters[p_idx[0]].size() + n_clusters[n_idx[0]].size()) * r;

                if (fitErr < 10 / r) {
                    candidates.emplace_back(pi, n_idx[0]);
                    candidateCenters.push_back(center);
                    candidatesRadius.push_back(r);
                }
            }
        }
    }

    // draw candidate circle
    for (int i = 0; i < candidates.size(); ++i) {
        cv::Point loc(candidateCenters[i][0], candidateCenters[i][1]);
        cv::circle(image_, loc, candidatesRadius[i], cv::Vec3b(0, 255, 0));
    }

    /** use prior pattern to match the candidate **/
    std::vector<cv::Point2f> points, outCenters;
    for (const Eigen::Vector2d &center: candidateCenters) {
        points.emplace_back(center[0], center[1]);
    }
    // TODO: 1. improve the sensitivity to additional nearby outlier
    //  2. we should assume there are absence,
    //  (even wrong pairs inside the pattern (seems too hard, maybe solved previously by more samples)),
    //  also the centers are not accurate.
    //  (solution) maybe optimization liangzu;
    //  or maybe solved by mul-spline segment(just ignore the error segment) and ignore with go on;
    //  user interaction (note: find sample in raw data)
    bool isFound = cv::findCirclesGrid(points, cv::Size(pattern_->cols, pattern_->rows), outCenters,
                                       cv::CALIB_CB_ASYMMETRIC_GRID);
    if (!isFound)
        isFound = cv::findCirclesGrid(points, cv::Size(pattern_->cols, pattern_->rows), outCenters,
                                      cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING);
    /*if (isFound)
        circleExtractionImage = image_.clone();*/
    drawChessboardCorners(image_, cv::Size(pattern_->cols, pattern_->rows), cv::Mat(outCenters), isFound);
    if (isFound) {
        KDTreeVectorOfVectorsAdaptor<vectorofEigenMatrix<Eigen::Vector2d>, double, 2, nanoflann::metric_L2_Simple>
                kdTree(2, candidateCenters, 10);
        std::vector<size_t> orderIdxs(pattern_->rows * pattern_->cols);
        std::vector<double> out_dist_sqr(1);
        for (int i = 0; i < outCenters.size(); ++i) {
            Eigen::Vector2d c(outCenters[i].x, outCenters[i].y);
            kdTree.query(c.data(), 1, orderIdxs.data() + i, out_dist_sqr.data());
        }

        // features_
        for (auto idx: orderIdxs) {
            features_.push_back(std::make_shared<CalibCircle>(candidateCenters[idx], candidatesRadius[idx]));
        }

        //detectionImage = image_.clone();
    }

    return isFound;
}

void opengv2::CirclesEventFrame::fitCircle(const std::vector<uint> &pSet,
                                           const std::vector<uint> &nSet,
                                           Eigen::Ref<Eigen::Vector2d> center, double &radius) {
    double sum_x = 0, sum_y = 0;
    double sum_xx = 0, sum_yy = 0, sum_xy = 0;
    double sum_xxx = 0, sum_yyy = 0, sum_xyy = 0, sum_xxy = 0;
    for (auto p_idx: pSet) {
        const Eigen::Vector2d &sample = positiveEvents_[p_idx];

        sum_x += sample[0];
        sum_y += sample[1];

        double xx = sample[0] * sample[0];
        double yy = sample[1] * sample[1];
        double xy = sample[0] * sample[1];

        sum_xx += xx;
        sum_yy += yy;
        sum_xy += xy;

        sum_xxx += xx * sample[0];
        sum_yyy += yy * sample[1];
        sum_xyy += xy * sample[1];
        sum_xxy += sample[0] * xy;
    }
    for (auto n_idx: nSet) {
        const Eigen::Vector2d &sample = negativeEvents_[n_idx];

        sum_x += sample[0];
        sum_y += sample[1];

        double xx = sample[0] * sample[0];
        double yy = sample[1] * sample[1];
        double xy = sample[0] * sample[1];

        sum_xx += xx;
        sum_yy += yy;
        sum_xy += xy;

        sum_xxx += xx * sample[0];
        sum_yyy += yy * sample[1];
        sum_xyy += xy * sample[1];
        sum_xxy += sample[0] * xy;
    }

    Eigen::Matrix3d A;
    A << 2 * sum_x, 2 * sum_y, pSet.size() + nSet.size(),
            2 * sum_xx, 2 * sum_xy, sum_x,
            2 * sum_xy, 2 * sum_yy, sum_y;
    Eigen::Vector3d b(sum_xx + sum_yy, sum_xxx + sum_xyy, sum_xxy + sum_yyy);

    Eigen::Vector3d x = A.lu().solve(b);
    center = x.block<2, 1>(0, 0);
    radius = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2]);
}

bool
opengv2::CirclesEventFrame::rectifyFeatures(const std::unordered_set<int> &outlierIdxs,
                                            const Eigen::Ref<const Eigen::Matrix3d> &Rcw,
                                            const Eigen::Ref<const Eigen::Vector3d> &tcw) {
    KDTreeVectorOfVectorsAdaptor<vectorofEigenMatrix<Eigen::Vector2d>, double, 2, nanoflann::metric_L2_Simple>
            p_kdTree(2, positiveEvents_, 10), n_kdTree(2, negativeEvents_, 10);

    Eigen::Matrix3d Rsw = Rcw;
    Eigen::Vector3d tsw = tcw;
    auto camera = dynamic_cast<PinholeCamera *>(sensor_.get());
    cv::Mat tvec, rvec, Rvec, distCoeffs, cameraMatrix;
    cv::eigen2cv(Rsw, Rvec);
    cv::Rodrigues(Rvec, rvec);
    cv::eigen2cv(tsw, tvec);
    cv::eigen2cv(camera->distCoeffs(), distCoeffs);
    cv::eigen2cv(camera->K(), cameraMatrix);
    for (int kIdx = 0; kIdx < features_.size(); kIdx++) {
        auto f = dynamic_cast<CalibCircle *>(features_[kIdx].get());
        auto lm = f->landmark();
        Eigen::Vector3d center = lm->position();

        std::vector<cv::Point3f> objectPoints;
        objectPoints.reserve(5);
        objectPoints.emplace_back(center[0], center[1], center[2]);
        double skewR = pattern_->circleRadius / std::sqrt(2);
        // Four quadrant, considering circle distortion
        objectPoints.emplace_back(center[0] + skewR, center[1] + skewR, center[2]);
        objectPoints.emplace_back(center[0] + skewR, center[1] - skewR, center[2]);
        objectPoints.emplace_back(center[0] - skewR, center[1] - skewR, center[2]);
        objectPoints.emplace_back(center[0] - skewR, center[1] + skewR, center[2]);

        std::vector<cv::Point2f> imagePoints;
        cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
        vectorofEigenMatrix<Eigen::Vector2d> eigenImagePoints;
        eigenImagePoints.reserve(5);
        for (const auto &p: imagePoints) {
            eigenImagePoints.emplace_back(p.x, p.y);
        }

        // if center exceed image bound, delete feature
        if (imagePoints[0].x >= camera->size()[0] || imagePoints[0].y >= camera->size()[1] ||
            imagePoints[0].x < 0 || imagePoints[0].y < 0) {
            features_[kIdx] = nullptr;
            continue;
        }

        std::vector<double> radius;
        double maxRadius = 0;
        radius.reserve(4);
        for (int i = 1; i < 5; ++i) {
            radius.push_back((eigenImagePoints[i] - eigenImagePoints[0]).norm());
            if (radius.back() > maxRadius) {
                maxRadius = radius.back();
            }
        }

        double inlierThreshold = 3; // pixel unit
        std::vector<std::pair<size_t, double>> p_IndicesDists, n_IndicesDists;
        p_kdTree.index->radiusSearch(eigenImagePoints[0].data(), std::pow(maxRadius + inlierThreshold, 2),
                                     p_IndicesDists,
                                     nanoflann::SearchParams(32, 0, false));
        n_kdTree.index->radiusSearch(eigenImagePoints[0].data(), std::pow(maxRadius + inlierThreshold, 2),
                                     n_IndicesDists,
                                     nanoflann::SearchParams(32, 0, false));

        // collect inliers
        std::pair<std::vector<uint>, std::vector<uint>> patternCircle;
        for (const auto &pair: p_IndicesDists) {
            Eigen::Vector2d direction = positiveEvents_[pair.first] - eigenImagePoints[0];
            double distance = std::sqrt(pair.second);
            int idx = 0;
            if (direction[0] >= 0 && direction[1] >= 0) {
                idx = 0;
            } else if (direction[0] >= 0 && direction[1] <= 0) {
                idx = 1;
            } else if (direction[0] <= 0 && direction[1] <= 0) {
                idx = 2;
            } else if (direction[0] <= 0 && direction[1] >= 0) {
                idx = 3;
            }

            if (std::abs(distance - radius[idx]) <= inlierThreshold) {
                patternCircle.first.push_back(pair.first);
            }
        }
        for (const auto &pair: n_IndicesDists) {
            Eigen::Vector2d direction = negativeEvents_[pair.first] - eigenImagePoints[0];
            double distance = std::sqrt(pair.second);
            int idx = 0;
            if (direction[0] >= 0 && direction[1] >= 0) {
                idx = 0;
            } else if (direction[0] >= 0 && direction[1] <= 0) {
                idx = 1;
            } else if (direction[0] <= 0 && direction[1] <= 0) {
                idx = 2;
            } else if (direction[0] <= 0 && direction[1] >= 0) {
                idx = 3;
            }

            if (std::abs(distance - radius[idx]) <= inlierThreshold) {
                patternCircle.second.push_back(pair.first);
            }
        }

        // expand sample set by clustering
        std::set<uint> pClusterSet, nClusterSet;
        std::unordered_map<uint, uint> pSample2Set, nSample2Set;
        for (int i = 0; i < pClusters_.size(); ++i) {
            for (int j = 0; j < pClusters_[i].size(); ++j) {
                pSample2Set.emplace(pClusters_[i][j], i);
            }
        }
        for (int i = 0; i < nClusters_.size(); ++i) {
            for (int j = 0; j < nClusters_[i].size(); ++j) {
                nSample2Set.emplace(nClusters_[i][j], i);
            }
        }
        for (const auto &p : patternCircle.first) {
            if (pSample2Set.find(p) != pSample2Set.end())
                pClusterSet.insert(pSample2Set[p]);
        }
        for (const auto &n : patternCircle.second) {
            if (nSample2Set.find(n) != nSample2Set.end())
                nClusterSet.insert(nSample2Set[n]);
        }
        patternCircle.first.clear();
        for (const auto &pCluster: pClusterSet) {
            for (const auto &p: pClusters_[pCluster]) {
                patternCircle.first.push_back(p);
            }
        }
        patternCircle.second.clear();
        for (const auto &nCluster: nClusterSet) {
            for (const auto &n: nClusters_[nCluster]) {
                patternCircle.second.push_back(n);
            }
        }

        // remove feature with too few measurements
        if (patternCircle.first.size() < 5 || patternCircle.second.size() < 5) {
            features_[kIdx] = nullptr;
            continue;
        }

        Eigen::Vector2d rectifiedCenter;
        double rectifiedRadius;
        fitCircle(patternCircle.first, patternCircle.second, rectifiedCenter,
                  rectifiedRadius);

        std::nth_element(radius.begin(), radius.begin() + radius.size() / 2, radius.end());
        // if fit result not good, delete feature
        if ((rectifiedCenter - eigenImagePoints[0]).norm() > 2 * inlierThreshold ||
            std::abs(rectifiedRadius - radius[radius.size() / 2]) > 1.5 * inlierThreshold) {
            features_[kIdx] = nullptr;
            continue;
        }

        f->setLocation(rectifiedCenter);
        f->radius = rectifiedRadius;
    }

    // release no longer used member
    positiveEvents_.clear();
    negativeEvents_.clear();

    // outlier features in the edges are more important
    std::vector<std::unordered_set<int>> edgePattern(4);
    std::vector<int> score(4, 0);
    for (int i = 0; i < pattern_->cols; ++i) // 1st row
        edgePattern[0].insert(i);
    for (int i = (pattern_->rows - 1) * pattern_->cols; i < (pattern_->rows * pattern_->cols); ++i) // last row
        edgePattern[1].insert(i);
    for (int i = 0;
         i < (pattern_->rows * pattern_->cols); i += (pattern_->isAsymmetric ? 2 : 1) * pattern_->cols) // 1st col
        edgePattern[2].insert(i);
    for (int i = pattern_->isAsymmetric ? (2 * pattern_->cols - 1) : (pattern_->cols - 1);
         i < (pattern_->rows * pattern_->cols); i += (pattern_->isAsymmetric ? 2 : 1) * pattern_->cols) // last col
        edgePattern[3].insert(i);

    int counter = 0, index = 0;
    for (auto itr = features_.begin(); itr != features_.end(); index++) {
        if (*itr == nullptr) {
            itr = features_.erase(itr);
            for (int i = 0; i < 4; ++i) {
                if (edgePattern[i].find(index) != edgePattern[i].end())
                    score[i]++;
            }
            counter++;
        } else {
            itr++;
        }
    }

    // check features on edge
    if (!params_.fitCircle) {
        for (int i = 0; i < 4; ++i) {
            if (score[i] >= (int) (edgePattern[i].size() - 1))
                return false;
        }
    }

    // remove frame with too many features gone
    if (counter >= 0.2 * (pattern_->cols * pattern_->rows)) {
        return false;
    }

    for (auto fb: features_) {
        auto f = dynamic_cast<CalibCircle *>(fb.get());
        cv::Point loc(f->location()[0], f->location()[1]);
        cv::circle(image_, loc, f->radius, cv::Vec3b(255, 255, 255));
    }

    // Building KD-Tree for neighbor searching on features_, used for establish corresponds for given event.
    circles_.reserve(features_.size());
    for (const auto &f: features_) {
        circles_.push_back(f->location());
    }
    circleKdTree_ = std::make_shared<KDTreeVectorOfVectorsAdaptor<vectorofEigenMatrix<Eigen::Vector2d>, double, 2>>
            (2, circles_, 10);

    return true;
}