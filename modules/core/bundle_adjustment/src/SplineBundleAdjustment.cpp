//
// Created by huangkun on 2020/9/4.
//

#include <opengv2/bundle_adjustment/SplineBundleAdjustment.hpp>
#include <opengv2/frame/CameraFrame.hpp>

opengv2::SplineBundleAdjustment::SplineBundleAdjustment(bool fixTsb, bool robust, bool RtConstrain,
                                                        bool planarConstrain, double RtConstrainWeight)
        : fixTsb_(fixTsb), robust_(robust), RtConstrain_(RtConstrain),
          planarConstrain_(planarConstrain), RtConstrainWeight_(RtConstrainWeight) {}

void opengv2::SplineBundleAdjustment::run(const std::map<double, Bodyframe::Ptr> &keyframes,
                                          const std::map<int, LandmarkBase::Ptr> &landmarks) {
    std::vector < Eigen::Matrix < double, 7, 1 >, Eigen::aligned_allocator < Eigen::Matrix < double, 7, 1>>> Q;
    std::vector<double> u;
    for (const auto &itr: keyframes) {
        auto &bf = itr.second;
        Eigen::Quaterniond Qwb = bf->unitQwb();
        Eigen::Vector3d twb = bf->twb();
        Eigen::Quaterniond Qbw = Qwb.conjugate();

        Q.emplace_back();
        Q.back().head(4) = Qbw.coeffs();
        Q.back().tail(3) = twb;

        u.push_back(bf->timeStamp());
    }
    Tspline_ = BsplineReal<7>(3, Q, -1, u);
    std::cout << "BundleAdjustment: " << Tspline_.getCP().size() << " control points." << std::endl;

    ceres::Problem problem;
    ceres::LocalParameterization *pose_parameterization = planarConstrain_ ?
                                                          new ceres::ProductParameterization(
                                                                  new ceres::EigenQuaternionParameterization(),
                                                                  new ceres::SubsetParameterization(3, {2})) :
                                                          new ceres::ProductParameterization(
                                                                  new ceres::EigenQuaternionParameterization(),
                                                                  new ceres::IdentityParameterization(3));
    // spline control points
    for (Eigen::Matrix<double, 7, 1> &cp: Tspline_.getCP()) {
        problem.AddParameterBlock(cp.data(), 7, pose_parameterization);
    }
    problem.SetParameterBlockConstant(Tspline_.getCP().front().data());

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

    // extract spline pose
    for (const auto &itr: keyframes) {
        auto &bf = itr.second;

        std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> Ders;
        Tspline_.evaluate(bf->timeStamp(), 0, Ders);
        Ders[0].head(4).normalize();

        bf->optT.clear();
        std::copy(Ders[0].data(), Ders[0].data() + 7, std::back_inserter(bf->optT));
    }
}

void opengv2::SplineBundleAdjustment::optimize(ceres::Problem &problem,
                                               const std::map<double, Bodyframe::Ptr> &keyframes,
                                               const std::map<int, LandmarkBase::Ptr> &landmarks) {
    ceres::Solver::Options options;
    auto *ordering = new ceres::ParameterBlockOrdering;

    // Tcb
    for (Eigen::Matrix<double, 7, 1> &Tsb: Bodyframe::optTsb) {
        ordering->AddElementToGroup(Tsb.data(), 2);
    }

    // control points
    for (Eigen::Matrix<double, 7, 1> &cp: Tspline_.getCP()) {
        ordering->AddElementToGroup(cp.data(), 1);
    }

    // pre-caculate spline basis
    std::unordered_map<double, std::shared_ptr<std::vector<std::vector<double>>>> basisFuns;
    std::unordered_map<double, size_t> spanIdxs;
    for (const auto &itr: keyframes) {
        auto &bf = itr.second;

        auto basisFun = std::make_shared<std::vector<std::vector<double>>>();
        size_t spanIdx = Tspline_.findSpan(bf->timeStamp());
        Tspline_.dersBasisFuns(bf->timeStamp(), spanIdx, 1, *basisFun);

        basisFuns[bf->timeStamp()] = basisFun;
        spanIdxs[bf->timeStamp()] = spanIdx;
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
            size_t spanIdx = spanIdxs[bf->timeStamp()];
            bool isMono = loc.size() == 2;
            ceres::CostFunction *cost_function = ReprojectionError::Create(
                    feature->bearingVector(), basisFuns[bf->timeStamp()], isMono);
            ceres::LossFunction *loss_function = new ceres::ScaledLoss(robust_ ? (isMono ? huber2D : huber3D) : nullptr,
                                                                       1,
                                                                       ceres::DO_NOT_TAKE_OWNERSHIP);
            problem.AddResidualBlock(cost_function, loss_function,
                                     Tspline_.getCP()[spanIdx - 3 + 0].data(),
                                     Tspline_.getCP()[spanIdx - 3 + 1].data(),
                                     Tspline_.getCP()[spanIdx - 3 + 2].data(),
                                     Tspline_.getCP()[spanIdx - 3 + 3].data(),
                                     lm->optData.data(), bf->optTsb[obs.frameId()].data());
        }
        lm->observationMutex.unlock_shared();
    }

    // constrain between rotation and translation
    if (RtConstrain_) {
        ceres::LossFunction *loss_function = new ceres::ScaledLoss(nullptr, RtConstrainWeight_,
                                                                   ceres::DO_NOT_TAKE_OWNERSHIP);
        for (const auto &itr: keyframes) {
            auto &bf = itr.second;
            size_t spanIdx = spanIdxs[bf->timeStamp()];

            ceres::CostFunction *cost_function = ConstrainTcbConstant::Create(basisFuns[bf->timeStamp()]);
            problem.AddResidualBlock(cost_function, loss_function,
                                     Tspline_.getCP()[spanIdx - 3 + 0].data(),
                                     Tspline_.getCP()[spanIdx - 3 + 1].data(),
                                     Tspline_.getCP()[spanIdx - 3 + 2].data(),
                                     Tspline_.getCP()[spanIdx - 3 + 3].data());
        }
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