//
// Created by huangkun on 2020/9/2.
//

#include <opengv2/bundle_adjustment/NormalBundleAdjustment.hpp>
#include <opengv2/frame/CameraFrame.hpp>

opengv2::NormalBundleAdjustment::NormalBundleAdjustment(bool fixTsb, bool robust, bool planarConstrain)
        : fixTsb_(fixTsb), robust_(robust), planarConstrain_(planarConstrain) {}

void opengv2::NormalBundleAdjustment::run(const std::map<double, Bodyframe::Ptr> &keyframes,
                                          const std::map<int, LandmarkBase::Ptr> &landmarks) {
    ceres::Problem problem;
    ceres::LocalParameterization *pose_parameterization = planarConstrain_ ?
                                                          new ceres::ProductParameterization(
                                                                  new ceres::EigenQuaternionParameterization(),
                                                                  new ceres::SubsetParameterization(3, {2})) :
                                                          new ceres::ProductParameterization(
                                                                  new ceres::EigenQuaternionParameterization(),
                                                                  new ceres::IdentityParameterization(3));
    for (const auto &itr: keyframes) {
        auto &bf = itr.second;

        Eigen::Quaterniond Qwb = bf->unitQwb();
        Eigen::Vector3d twb = bf->twb();

        Eigen::Quaterniond Qbw = Qwb.conjugate();
        bf->optT.clear();
        std::copy(Qbw.coeffs().data(), Qbw.coeffs().data() + 4, std::back_inserter(bf->optT));
        std::copy(twb.data(), twb.data() + 3, std::back_inserter(bf->optT));

        problem.AddParameterBlock(bf->optT.data(), 7, pose_parameterization);
        if (bf->fixedInBA)
            problem.SetParameterBlockConstant(bf->optT.data());
    }
    problem.SetParameterBlockConstant(keyframes.begin()->second->optT.data());

    // landmarks
    for (const auto &itr: landmarks) {
        auto &lm = itr.second;
        Eigen::Vector3d p = lm->position();

        lm->optData.clear();
        std::copy(p.data(), p.data() + 3, std::back_inserter(lm->optData));

        problem.AddParameterBlock(lm->optData.data(), 3);
        if (lm->fixedInBA) {
            problem.SetParameterBlockConstant(lm->optData.data());
        }
    }

    // Tcb
    ceres::LocalParameterization *parameterization = new ceres::ProductParameterization(
            new ceres::EigenQuaternionParameterization(), new ceres::IdentityParameterization(3));
    Bodyframe::optTsb.clear();
    for (size_t i = 0; i < Bodyframe::size(); i++) {
        Bodyframe::optTsb.emplace_back();
        Bodyframe::optTsb[i].head<4>() = Bodyframe::unitQsb(i).coeffs();
        Bodyframe::optTsb[i].tail<3>() = Bodyframe::tsb(i);

        problem.AddParameterBlock(Bodyframe::optTsb[i].data(), 7, parameterization);
        if (fixTsb_)
            problem.SetParameterBlockConstant(Bodyframe::optTsb[i].data());
    }

    optimize(problem, keyframes, landmarks);
}

void opengv2::NormalBundleAdjustment::optimize(ceres::Problem &problem,
                                               const std::map<double, Bodyframe::Ptr> &keyframes,
                                               const std::map<int, LandmarkBase::Ptr> &landmarks) {
    ceres::Solver::Options options;
    auto *ordering = new ceres::ParameterBlockOrdering;

    // Tcb
    for (Eigen::Matrix<double, 7, 1> &Tsb: Bodyframe::optTsb) {
        ordering->AddElementToGroup(Tsb.data(), 2);
    }

    // frame
    for (const auto &itr: keyframes) {
        auto &bf = itr.second;
        ordering->AddElementToGroup(bf->optT.data(), 1);
    }

    ceres::LossFunction *huber2D = new ceres::HuberLoss(5. / 800.);
    ceres::LossFunction *huber3D = new ceres::HuberLoss(5. / 800. * 10);// TODO: find suitable huber bound
    for (const auto &itr: landmarks) {
        auto &lm = itr.second;
        ordering->AddElementToGroup(lm->optData.data(), 0);

        lm->observationMutex.lock_shared();
        for (const auto &itr1: lm->observations()) {
            const auto &obs = itr1.second;
            auto result = keyframes.find(obs.timestamp());
            Bodyframe::Ptr bf = result == keyframes.end() ? nullptr : result->second;
            if (bf == nullptr || bf->frame(obs.frameId())->sensor()->type() != SensorType::Camera)
                continue;
            auto feature = obs.feature();
            const Eigen::VectorXd &loc = feature->location();
            bool isMono = loc.size() == 2;
            ceres::CostFunction *cost_function = ReprojectionError::Create(feature->bearingVector(), isMono);
            ceres::LossFunction *loss_function = new ceres::ScaledLoss(robust_ ? (isMono ? huber2D : huber3D) : nullptr,
                                                                       1,
                                                                       ceres::DO_NOT_TAKE_OWNERSHIP);
            problem.AddResidualBlock(cost_function, loss_function,
                                     bf->optT.data(), lm->optData.data(), bf->optTsb[obs.frameId()].data());
        }
        lm->observationMutex.unlock_shared();
    }

    options.linear_solver_ordering.reset(ordering);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    //options.minimizer_progress_to_stdout = true;
    options.num_linear_solver_threads = 3;
    options.num_threads = 3;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Use FullReport() to diagnose performance problems
    std::cout << summary.BriefReport() << std::endl;
}