//
// Created by huangkun on 2020/9/4.
//

#include <opengv2/bundle_adjustment/RelativeConstraintBA.hpp>
#include <opengv2/frame/CameraFrame.hpp>

opengv2::RelativeConstraintBA::RelativeConstraintBA(bool fixTsb, bool robust, bool planarConstrain, double weight)
        : NormalBundleAdjustment(fixTsb, robust, planarConstrain), weight_(weight) {}

void opengv2::RelativeConstraintBA::optimize(ceres::Problem &problem,
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

    // Relative constraint
    ceres::LossFunction *loss_function = new ceres::ScaledLoss(nullptr, weight_, ceres::DO_NOT_TAKE_OWNERSHIP);
    for (auto itr0 = keyframes.begin(), itr1 = ++(keyframes.begin());
         itr1 != keyframes.end(); itr0 = (itr1++)) {
        auto bf0 = itr0->second;
        auto bf1 = itr1->second;
        ceres::CostFunction *cost_function = RelativeConstraint::Create();
        problem.AddResidualBlock(cost_function, loss_function, bf0->optT.data(), bf1->optT.data());
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