//
// Created by huangkun on 2020/9/4.
//

#include <opengv2/bundle_adjustment/SplineBundleAdjustmentV2.hpp>
#include <opengv2/frame/CameraFrame.hpp>

opengv2::SplineBundleAdjustmentV2::SplineBundleAdjustmentV2(bool fixTsb, bool robust) : fixTsb_(fixTsb),
                                                                                        robust_(robust) {}

void opengv2::SplineBundleAdjustmentV2::run(const std::map<double, Bodyframe::Ptr> &keyframes,
                                            const std::map<int, LandmarkBase::Ptr> &landmarks) {
    std::vector <Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> Q;
    std::vector<double> u;
    for (const auto &itr: keyframes) {
        auto &bf = itr.second;
        Eigen::Quaterniond Qwb = bf->unitQwb();
        Eigen::Vector3d twb = bf->twb();

        double theta;
        Eigen::Vector3d d;
        Eigen::Matrix3d Rbw = Qwb.conjugate().matrix();
        rotMatrixToDer(Rbw, d, theta);
        Q.emplace_back();
        Q.back() << twb[0], twb[1], twb[2], theta;

        u.push_back(bf->timeStamp());
    }
    Tspline_ = BsplineReal<4>(3, Q, -1, u);
    std::cout << "BundleAdjustment: " << Tspline_.getCP().size() << " control points." << std::endl;

    ceres::Problem problem;
    // spline control points
    for (Eigen::Matrix<double, 4, 1> &cp: Tspline_.getCP()) {
        problem.AddParameterBlock(cp.data(), 4);
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

        std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>> Ders;
        Tspline_.evaluate(bf->timeStamp(), 1, Ders);
        Eigen::Vector3d twb(Ders[0][0], Ders[0][1], Ders[0][2]);
        Eigen::Vector3d y(Ders[1][0], Ders[1][1], Ders[1][2]);
        y.normalize();
        Eigen::Matrix3d Rbw;
        derToRotMatrix(y, Ders[0][3], Rbw);
        Eigen::Quaterniond Qbw(Rbw);
        Qbw.normalize();

        bf->optT.clear();
        std::copy(Qbw.coeffs().data(), Qbw.coeffs().data() + 4, std::back_inserter(bf->optT));
        std::copy(twb.data(), twb.data() + 3, std::back_inserter(bf->optT));
    }
}

void opengv2::SplineBundleAdjustmentV2::optimize(ceres::Problem &problem,
                                                 const std::map<double, Bodyframe::Ptr> &keyframes,
                                                 const std::map<int, LandmarkBase::Ptr> &landmarks) {
    ceres::Solver::Options options;
    auto *ordering = new ceres::ParameterBlockOrdering;

    // Tcb
    for (Eigen::Matrix<double, 7, 1> &Tsb: Bodyframe::optTsb) {
        ordering->AddElementToGroup(Tsb.data(), 2);
    }

    // control points
    for (Eigen::Matrix<double, 4, 1> &cp: Tspline_.getCP()) {
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