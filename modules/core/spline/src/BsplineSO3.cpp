//
// Created by huangkun on 2020/8/26.
//

#include <Eigen/SparseCholesky>
#include <cmath>
#include <algorithm>
#include <numeric>

#include <opengv2/spline/BsplineSO3.hpp>

opengv2::BsplineSO3::BsplineSO3(int p,
                                const std::vector<Sophus::SO3d, Eigen::aligned_allocator<Sophus::SO3d>> &samples,
                                int controlPointsNum, const std::vector<double> &correspondingUs,
                                const std::vector<std::vector<Eigen::Vector3d>> &derivativeSamples,
                                double derivativeWeight) : degree_(p), derivativeWeight_(derivativeWeight) {
    // if dataPoints given, then do approximation
    if (!samples.empty()) {
        samples_ = samples;

        // Data registration
        if (!correspondingUs.empty()) {
            correspondingUs_ = correspondingUs;
        } else {
            // Data registration using chord length, Range: [0,1]
            correspondingUs_.resize(samples_.size());

            double d = 0;
            for (int i = 1; i < samples_.size(); ++i) {
                d += 2 * std::acos((samples_[i] * samples_[i - 1].inverse()).unit_quaternion().w());
            }

            correspondingUs_.front() = 0;
            correspondingUs_.back() = 1;
            for (int i = 1; i < samples_.size() - 1; ++i) {
                correspondingUs_[i] = correspondingUs_[i - 1] +
                                      2 * std::acos((samples_[i] * samples_[i - 1].inverse()).unit_quaternion().w()) /
                                      d;
            }
        }

        // derivative
        if (!derivativeSamples.empty()) {
            derivativeSamples_ = derivativeSamples;
        }

        // initialization
        if (controlPointsNum < 0)
            controlPointsNum = std::max(int(samples_.size() / 3), degree_ + 1);
        if (controlPointsNum <= degree_) {
            std::cerr << "BsplineSO3 Constructor: control points number should greater than degree!" << std::endl;
            return;
        }
        knotSpacing(controlPointsNum);
        initialGuess();
        optimizeCP();
    }
}

void opengv2::BsplineSO3::knotSpacing(int controlPointsNum) {
    /*** knot initialization (9.68) in NURBS book ***/
    knotVector_.resize(controlPointsNum + degree_ + 1);
    std::fill_n(knotVector_.begin(), degree_ + 1, correspondingUs_.front());
    std::fill_n(knotVector_.rbegin(), degree_ + 1, correspondingUs_.back());
    double d = samples_.size() / double(controlPointsNum - degree_);
    for (int j = 1; j <= controlPointsNum - 1 - degree_; j++) {
        int i = floor(j * d);
        double alpha = j * d - i;
        knotVector_[degree_ + j] = (1 - alpha) * correspondingUs_[i - 1] + alpha * correspondingUs_[i];
    }
}

void opengv2::BsplineSO3::derBasisFuns(double u, size_t spanIdx, int derivativeLimit,
                                       std::vector<std::vector<double>> &ders) const {
    // check
    if (knotVector_.empty()) {
        std::cerr << "Function derBasisFuns: knotVector is empty!" << std::endl;
        return;
    }

    ders.resize(derivativeLimit + 1);
    for (auto itr = ders.begin(); itr != ders.end(); itr++) {
        itr->assign(degree_, 0);
    }

    std::vector<std::vector<double>> basisN;

    // 0-th derivative: $\beta_{k,i}(u) = \sum_{j=i}^{k} N_{j,p}(u), i \in [k-p+1, k]$.
    // since \sum_{j=k-p}^{k} N_{j,p}(u) = 1
    basis(u, spanIdx, 0, degree_, basisN);
    ders[0][degree_ - 1] = basisN[0][degree_];
    for (int i = degree_ - 2; i >= 0; i--) {
        ders[0][i] = ders[0][i + 1] + basisN[0][i + 1];
    }

    // \alpha-th derivative, \alpha >= 1
    if (derivativeLimit > 0) {
        basisN.clear();
        basis(u, spanIdx, derivativeLimit - 1, degree_ - 1, basisN);

        for (int alpha = 1; alpha <= derivativeLimit; alpha++) {
            for (int i = spanIdx - degree_ + 1; i <= spanIdx; i++) {
                ders[alpha][i - (spanIdx - degree_ + 1)] =
                        degree_ / (knotVector_[i + degree_] - knotVector_[i]) *
                        basisN[alpha][i - (spanIdx - degree_ + 1)];
            }
        }
    }
}

void opengv2::BsplineSO3::basis(double u, size_t spanIdx, int derivativeLimit, int degree,
                                std::vector<std::vector<double>> &ders) const {
    ders.resize(derivativeLimit + 1);
    for (auto &it:ders) {
        it.assign(degree + 1, 0);
    }

    double ndu[degree + 1][degree + 1]; // store the basis functions and knot differences

    // store (in an alternating fashion) the two most recently computed rows a_{k,j} and a_{k-1,j}
    double a[2][degree + 1];

    std::vector<double> left, right;
    left.resize(degree + 1);
    right.resize(degree + 1);

    ndu[0][0] = 1;
    for (int j = 1; j <= degree; j++) {
        left[j] = u - knotVector_[spanIdx + 1 - j];
        right[j] = knotVector_[spanIdx + j] - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            ndu[j][r] = right[r + 1] + left[j - r];
            double temp = ndu[r][j - 1] / ndu[j][r];

            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    /* Load the basis functions */
    for (int j = 0; j <= degree; j++)
        ders[0][j] = ndu[j][degree];

    if (derivativeLimit > 0) {
        /*** This section computes the derivatives (Eq.[2.9]) ***/
        /* Load over function index */
        for (int r = 0; r <= degree; r++) {
            int s1 = 0, s2 = 1; // Alternate rows in array a
            a[0][0] = 1;

            // loop to compute kth derivative
            for (int k = 1; k <= derivativeLimit; k++) {
                double d = 0;
                int rk = r - k, pk = degree - k;
                if (r >= k) {
                    a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                    d = a[s2][0] * ndu[rk][pk];
                }

                int j1, j2;
                if (rk >= -1)
                    j1 = 1;
                else
                    j1 = -rk;

                if (r - 1 <= pk)
                    j2 = k - 1;
                else
                    j2 = degree - r;

                for (int j = j1; j <= j2; j++) {
                    a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                    d += a[s2][j] * ndu[rk + j][pk];
                }

                if (r <= pk) {
                    a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                    d += a[s2][k] * ndu[r][pk];
                }

                ders[k][r] = d;

                // switch rows
                std::swap(s1, s2);
            }
        }

        /* Multiply through by correct factors (Eq. [2.9]) */
        int r = degree;
        for (int k = 1; k <= derivativeLimit; ++k) {
            for (int j = 0; j <= degree; j++)
                ders[k][j] *= r;

            r *= (degree - k);
        }
    }
}

// TODO: an acceleration penalty factor
void opengv2::BsplineSO3::initialGuess() {
    /*** check ***/
    if (samples_.empty()) {
        std::cerr << "Function initialGuess: dataPoints is empty!" << std::endl;
        return;
    }
    if (knotVector_.empty()) {
        std::cerr << "Function initialGuess: knotVector is empty!" << std::endl;
        return;
    }

    int controlPointsNum = knotVector_.size() - degree_ - 1;

    // initialize controlPoints
    controlPoints_.resize(controlPointsNum);
    controlPoints_.front() = samples_.front();
    controlPoints_.back() = samples_.back();

    /*** Problem ***/
    Eigen::SparseMatrix<double> N(samples_.size(), controlPointsNum);
    Eigen::SparseMatrix<double> M(samples_.size(), controlPointsNum);
    std::vector<std::vector<double>> ders;
    for (int k = 0; k < samples_.size(); ++k) {
        size_t spanIdx = findSpan(correspondingUs_[k]);
        basis(correspondingUs_[k], spanIdx, 0, degree_, ders);

        for (int j = 0; j <= degree_; ++j) {
            if (ders[0][j] != 0)
                N.insert(k, spanIdx - degree_ + j) = ders[0][j]; // N_{i-p+j,p}^{0}(\bar{u}_k)
        }
    }

    // formulating B
    std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>> B;
    B.assign(controlPointsNum - 2, Eigen::Matrix<double, 4, 1>::Zero());
    for (int l = 1; l <= controlPointsNum - 2; ++l) {
        // for l-th col of N
        for (Eigen::SparseMatrix<double>::InnerIterator it(N, l); it; ++it) {
            // N_{l,p}(\bar{u}_k), valid for column-major
            int k = it.row();
            if (k != 0 && k != samples_.size() - 1) {
                B[l - 1] += it.value() * (samples_[k].unit_quaternion().coeffs() -
                                          N.coeff(k, 0) * samples_.front().unit_quaternion().coeffs() -
                                          N.coeff(k, controlPointsNum - 1) *
                                          samples_.back().unit_quaternion().coeffs());
            }
        }
    }

    // formulating M,N
    Eigen::SparseMatrix<double> Nc = N.block(1, 1, samples_.size() - 2, controlPointsNum - 2);
    Eigen::SparseMatrix<double> A = Nc.transpose() * Nc;

    // solving
    A.makeCompressed();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Function initialGuess: decomposition failed!" << std::endl;
        return;
    }
    Eigen::VectorXd b(controlPointsNum - 2);

    std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>> cp;
    cp.resize(controlPointsNum - 2);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < B.size(); ++j) {
            b[j] = B[j][i];
        }
        Eigen::VectorXd x = solver.solve(b);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Function initialGuess: solving failed!" << std::endl;
            return;
        }
        for (int j = 1; j <= controlPointsNum - 2; ++j) {
            cp[j - 1][i] = x[j - 1];
        }
    }
    for (int j = 1; j <= controlPointsNum - 2; ++j) {
        controlPoints_[j].setQuaternion(Eigen::Quaterniond(cp[j - 1]));
    }
}

void opengv2::BsplineSO3::optimizeCP() {
    ceres::Problem problem;
    ceres::LocalParameterization *SO3_parameterization = new LocalParameterizationSO3();

    // Specify local update rule for our parameter
    for (Sophus::SO3d &cp:controlPoints_) {
        problem.AddParameterBlock(cp.data(), Sophus::SO3d::num_parameters, SO3_parameterization);
    }
    problem.SetParameterBlockConstant(controlPoints_.front().data());
    problem.SetParameterBlockConstant(controlPoints_.back().data());

    // pre-calculate spline basis
    std::unordered_map<int, std::shared_ptr<std::vector<std::vector<double>>>> basisFuns;
    std::unordered_map<int, size_t> spanIdxs;
    for (int i = 0; i < samples_.size(); ++i) {
        double u = correspondingUs_[i];
        auto basisFun = std::make_shared<std::vector<std::vector<double>>>();
        size_t spanIdx = findSpan(u);
        derBasisFuns(u, spanIdx, 0, *basisFun);

        basisFuns[i] = basisFun;
        spanIdxs[i] = spanIdx;
    }

    // Create and add cost functions. Derivatives will be evaluated via automatic differentiation
    for (int i = 0; i < samples_.size(); ++i) {
        auto spanIdx = spanIdxs[i];
        ceres::CostFunction *cost_function;
        if (degree_ == 4) {
            cost_function = P4ApproximationError::Create(samples_[i].inverse(), basisFuns[i]);
            problem.AddResidualBlock(cost_function, nullptr,
                                     controlPoints_[spanIdx - 4 + 0].data(),
                                     controlPoints_[spanIdx - 4 + 1].data(),
                                     controlPoints_[spanIdx - 4 + 2].data(),
                                     controlPoints_[spanIdx - 4 + 3].data(),
                                     controlPoints_[spanIdx - 4 + 4].data());
        } else if (degree_ == 3) {
            cost_function = P3ApproximationError::Create(samples_[i].inverse(), basisFuns[i]);
            problem.AddResidualBlock(cost_function, nullptr,
                                     controlPoints_[spanIdx - 3 + 0].data(),
                                     controlPoints_[spanIdx - 3 + 1].data(),
                                     controlPoints_[spanIdx - 3 + 2].data(),
                                     controlPoints_[spanIdx - 3 + 3].data());
        }
    }

    // Set solver options (precision / method)
    ceres::Solver::Options options;
    options.gradient_tolerance = Sophus::Constants<double>::epsilon();
    options.function_tolerance = Sophus::Constants<double>::epsilon();
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    // Solve
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
}

void opengv2::BsplineSO3::evaluate(double u, int derivativeLimit, Sophus::SO3d &val,
                                   std::vector<Eigen::Vector3d> &Ders) const {
    if (controlPoints_.empty()) {
        std::cerr << "Function evaluate: controlPoints is empty!" << std::endl;
        return;
    }
    if (derivativeLimit > 2) {
        std::cerr << "Function evaluate: derivativeLimit should <= 2!" << std::endl;
        return;
    }
    Ders.clear();

    // compute basis, [k-p+1,k]
    std::vector<std::vector<double>> basis;
    size_t spanIdx = findSpan(u);
    derBasisFuns(u, spanIdx, derivativeLimit, basis);

    // d_{k-p+j}, j \in [1,p]
    std::vector<Eigen::Vector3d> d; // since Sophus::SO3d::DoF = 3
    for (int j = 1; j <= degree_; ++j) {
        d.push_back(
                (controlPoints_[spanIdx - degree_ + j - 1].inverse() * controlPoints_[spanIdx - degree_ + j]).log());
    }

    // A_j(u), j \in [1,p]
    std::vector<Sophus::SO3d, Eigen::aligned_allocator<Sophus::SO3d>> A;
    for (int j = 1; j <= degree_; ++j) {
        int j_idx = j - 1;
        A.push_back(Sophus::SO3d::exp(basis[0][j_idx] * d[j_idx]));
    }

    // derivative = 0
    val = controlPoints_[spanIdx - degree_ + 0];
    for (int j = 1; j <= degree_; ++j) {
        int j_idx = j - 1;
        val *= A[j_idx];
    }

    if (derivativeLimit >= 1) {
        std::vector<std::vector<Eigen::Vector3d>> angVDer; // since Sophus::SO3d::DoF = 3
        angVDer.resize(derivativeLimit);
        for (auto &ad: angVDer) {
            ad.assign(degree_ + 1, Eigen::Vector3d::Zero());
        }

        for (int j = 2; j <= degree_ + 1; j++) {
            int j_idx = j - 1;
            angVDer[0][j_idx] =
                    A[j_idx - 1].inverse().Adj() * angVDer[0][j_idx - 1] + basis[1][j_idx - 1] * d[j_idx - 1];
        }

        Ders.push_back(angVDer[0].back());

        if (derivativeLimit >= 2) {
            for (int j = 2; j <= degree_ + 1; j++) {
                int j_idx = j - 1;
                angVDer[1][j_idx] = basis[1][j_idx - 1] * Sophus::SO3d::lieBracket(angVDer[0][j_idx], d[j_idx - 1]) +
                                    A[j_idx - 1].inverse().Adj() * angVDer[1][j_idx - 1] +
                                    basis[2][j_idx - 1] * d[j_idx - 1];
            }

            Ders.push_back(angVDer[1].back());
        }
    }
}
