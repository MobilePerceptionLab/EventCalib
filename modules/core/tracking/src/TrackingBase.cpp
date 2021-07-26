//
// Created by huangkun on 2020/2/12.
//

#include <opengv2/tracking/TrackingBase.hpp>
#include <opengv2/frame/CameraFrame.hpp>
#include <opengv2/map/MapBase.hpp>

opengv2::TrackingBase::TrackingBase(MapBase::Ptr map)
        : state(NOT_INITIALIZED), map_(std::move(map)), system_(nullptr), landmarkCounter_(0) {}

void opengv2::TrackingBase::setSystem(SystemBase *system) {
    system_ = system;
}

bool opengv2::TrackingBase::process(Bodyframe::Ptr bodyframe) {
    stateMutex.lock_shared();
    if (state == NOT_INITIALIZED) {
        stateMutex.unlock_shared();

        stateMutex.lock();
        if (state == NOT_INITIALIZED) {
            bool ok = initialization(bodyframe); // execute only once
            if (ok)
                state = OK;
            stateMutex.unlock();
            return ok;
        } else {
            stateMutex.unlock();
        }
    } else {
        stateMutex.unlock_shared();
    }

    stateMutex.lock_shared();
    if (state == OK) {
        stateMutex.unlock_shared();
        // System is initialized. Track Frame.
        bool ok = track(bodyframe);
        return ok;
    } else {
        stateMutex.unlock_shared();
    }

    return false;
}

Eigen::Vector3d
opengv2::TrackingBase::triangulation(const Eigen::Quaterniond &Q12, const Eigen::Ref<const Eigen::Vector3d> &t12,
                                     const Eigen::Ref<const Eigen::Vector3d> &bv1,
                                     const Eigen::Ref<const Eigen::Vector3d> &bv2) {
    Eigen::Vector3d bv2_unrotated = Q12 * bv2;
    Eigen::Vector2d b;
    b[0] = t12.dot(bv1);
    b[1] = t12.dot(bv2_unrotated);
    Eigen::Matrix2d A;
    A(0, 0) = bv1.dot(bv1);
    A(1, 0) = bv1.dot(bv2_unrotated);
    A(0, 1) = -A(1, 0);
    A(1, 1) = -bv2_unrotated.dot(bv2_unrotated);
    Eigen::Vector2d lambda = A.inverse() * b;
    Eigen::Vector3d xm = lambda[0] * bv1;
    Eigen::Vector3d xn = t12 + lambda[1] * bv2_unrotated;
    return (xm + xn) / 2;
}

void opengv2::TrackingBase::createLandmark(opengv2::Bodyframe::Ptr bf1, opengv2::Bodyframe::Ptr bf2,
                                           std::vector <MatchBase::Ptr> &matches, bool enableCheck) {
    std::vector <Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> Qc1c2;
    std::vector <Eigen::Vector3d> tc1c2;
    for (int i = 0; i < Bodyframe::size(); ++i) {
        Eigen::Quaterniond Qwc1 = bf1->unitQwb() * Bodyframe::unitQsb(i).conjugate();
        Eigen::Vector3d twc1 =
                bf1->twb() - bf1->unitQwb() * (Bodyframe::unitQsb(i).conjugate() * Bodyframe::tsb(i));
        Eigen::Quaterniond Qwc2 = bf2->unitQwb() * Bodyframe::unitQsb(i).conjugate();
        Eigen::Vector3d twc2 =
                bf2->twb() - bf2->unitQwb() * (Bodyframe::unitQsb(i).conjugate() * Bodyframe::tsb(i));
        Qc1c2.push_back(Qwc1.conjugate() * Qwc2);
        Qc1c2.back().normalize();
        tc1c2.push_back(Qwc1.conjugate() * (twc2 - twc1));
    }


    for (auto match = matches.begin(); match != matches.end();) {
        auto match2D2D = dynamic_cast<Match2D2D *>((*match).get());
        if (match2D2D == nullptr) {// TODO: support for other match type
            match++;
            continue;
        }
        const auto &fi1 = match2D2D->fi1();
        const auto &fi2 = match2D2D->fi2();
        auto f1 = fi1.feature();
        auto f2 = fi2.feature();
        if (f1 == nullptr || f2 == nullptr) {
            match = matches.erase(match);
            continue;
        }
        auto lm1 = f1->landmark();
        auto lm2 = f2->landmark();
        const Eigen::Vector3d &bv1 = f1->bearingVector();
        const Eigen::Vector3d &bv2 = f2->bearingVector();
        Eigen::Vector3d Xc1 = triangulation(Qc1c2[fi1.frameId()], tc1c2[fi1.frameId()], bv1, bv2);
        if (Xc1.hasNaN()) {
            match = matches.erase(match);
            continue;
        }
        Eigen::Vector3d Xc2 = Qc1c2[fi1.frameId()].conjugate() * (Xc1 - tc1c2[fi1.frameId()]);

        // epipolar check (contain view angle check), maybe already contained in RANSAC
        if (1.0 - (bv1.transpose() * (Xc1 / Xc1.norm())) > 1 - cos(atan(3 / 800.)) ||
            1.0 - (bv2.transpose() * (Xc2 / Xc2.norm())) > 1 - cos(atan(3 / 800.))) {
            match = matches.erase(match);
            continue;
        }

        if (lm1 == nullptr) {
            if (lm2 == nullptr) { // create new landmark
                // depth check, only valid when the scale is correct
                if (enableCheck && (Xc1.norm() <= 0.3 || Xc1.norm() > 30 || Xc2.norm() <= 0.3 || Xc2.norm() > 30)) {
                    match = matches.erase(match);
                    continue;
                }

                Eigen::Vector3d Xb =
                        Bodyframe::unitQsb(fi1.frameId()).conjugate() * (Xc1 - Bodyframe::tsb(fi1.frameId()));
                Eigen::Vector3d Xw = bf1->unitQwb() * Xb + bf1->twb();

                LandmarkBase::Ptr lm = std::make_shared<LandmarkBase>(landmarkCounter_++, Xw);
                lm->addObservation(fi1);
                lm->addObservation(fi2);
                map_->addLandmark(lm);

                //add references in the two keyframes to this new landmark
                f1->setLandmark(lm);
                f2->setLandmark(lm);

                // only add once
                activeLandmarks_.push_back(lm->id());
            } else { // existing landmark in 2
                if (enableCheck) {// landmark projection
                    Eigen::Vector3d Xb1 = bf1->unitQwb().conjugate() * (lm2->position() - bf1->twb());
                    Xc1 = Bodyframe::unitQsb(fi1.frameId()) * Xb1 + Bodyframe::tsb(fi1.frameId());

                    // depth check, only valid when the scale is correct
                    if (Xc1.norm() <= 0.3 || Xc1.norm() > 30) {
                        match = matches.erase(match);
                        continue;
                    }

                    // landmark projection check, only valid when the scale is correct
                    if (1.0 - (bv1.transpose() * (Xc1 / Xc1.norm())) > 1 - cos(atan(5 / 800.))) {
                        match = matches.erase(match);
                        continue;
                    }
                }

                lm2->addObservation(fi1);
                f1->setLandmark(lm2);
            }
        } else {
            if (lm2 == nullptr) { // existing landmark in 1
                if (enableCheck) {// landmark projection
                    Eigen::Vector3d Xb2 = bf2->unitQwb().conjugate() * (lm1->position() - bf2->twb());
                    Xc2 = Bodyframe::unitQsb(fi2.frameId()) * Xb2 + Bodyframe::tsb(fi2.frameId());

                    // depth check, only valid when the scale is correct
                    if (Xc2.norm() <= 0.3 || Xc2.norm() > 30) {
                        match = matches.erase(match);
                        continue;
                    }

                    // landmark projection check, only valid when the scale is correct
                    if (1.0 - (bv2.transpose() * (Xc2 / Xc2.norm())) > 1 - cos(atan(5 / 800.))) {
                        match = matches.erase(match);
                        continue;
                    }
                }

                lm1->addObservation(fi2);
                f2->setLandmark(lm1);

            } else if (lm1->id() != lm2->id()) { // existing conflict landmark
                match = matches.erase(match);
                continue;
            }
        }

        match++;
    }
}
