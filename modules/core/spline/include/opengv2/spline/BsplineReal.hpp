#ifndef OPENGV2_BSPLINEREAL_HPP
#define OPENGV2_BSPLINEREAL_HPP

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace opengv2 {
    template<int dim>
    class BsplineReal {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        explicit BsplineReal(int p = dim,
                             const std::vector<Eigen::Matrix<double, dim, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, dim, 1>>> &Q = std::vector<Eigen::Matrix<double, dim, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, dim, 1>>>(),
                             int controlPointsNum = -1,
                             const std::vector<double> &u = std::vector<double>(),
                             const std::vector<Eigen::Matrix<double, dim, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, dim, 1>>> &dQ = std::vector<Eigen::Matrix<double, dim, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, dim, 1>>>(),
                         bool derWithMagnitude = false, double lambda = 1) : degree(p), lambda(lambda) {
            // if dataPoints given, then do approximation
            if (!Q.empty()) {
                dataPoints = Q;

                // Data registration
                if (!u.empty()) {
                    correspondingUs = u;
                } else {
                    // Data registration using chord length
                    // Range: [0,1]
                    correspondingUs.resize(dataPoints.size());

                    double d = 0;
                    for (int i = 1; i < dataPoints.size(); ++i) {
                        d += (dataPoints[i] - dataPoints[i - 1]).norm();
                    }

                    correspondingUs.front() = 0;
                    correspondingUs.back() = 1;
                    for (int i = 1; i < dataPoints.size() - 1; ++i) {
                        correspondingUs[i] =
                                correspondingUs[i - 1] + (dataPoints[i] - dataPoints[i - 1]).norm() / d;
                    }
                }

                // derivative
                if (!dQ.empty()) {
                    derivatives = dQ;

                    if (!derWithMagnitude) {
                        // derivatives is unit direction, magnitude estimation by speed estimation
                        for (int i = 1; i <= dataPoints.size() - 2; ++i) {
                            double speed = (dataPoints[i] - dataPoints[i - 1]).norm() /
                                           (correspondingUs[i] - correspondingUs[i - 1]) +
                                           (dataPoints[i + 1] - dataPoints[i]).norm() /
                                           (correspondingUs[i + 1] - correspondingUs[i]);
                            derivatives[i] *= speed / 2;
                        }
                        derivatives.front() *=
                                (dataPoints[1] - dataPoints[0]).norm() / (correspondingUs[1] - correspondingUs[0]);
                        derivatives.back() *=
                                (dataPoints[dataPoints.size() - 1] - dataPoints[dataPoints.size() - 2]).norm() /
                                (correspondingUs[dataPoints.size() - 1] - correspondingUs[dataPoints.size() - 2]);
                    }
                }

                // approximation
                if (controlPointsNum < 0)
                    controlPointsNum = std::max(int(Q.size() / 3), degree + 1);
                if (controlPointsNum <= degree) {
                    std::cerr << "Constructor: control points number should greater than degree!" << std::endl;
                    return;
                }
                approximation(controlPointsNum);
            }
        };


        /*
         * Approximation
         * return: status: failed(-1), success(0)
         */
        int approximation(int controlPointsNum) {
            /*** knot initialization (9.68) in NURBS book ***/
            knotVector.resize(controlPointsNum + degree + 1);
            std::fill_n(knotVector.begin(), degree + 1, correspondingUs.front());
            std::fill_n(knotVector.rbegin(), degree + 1, correspondingUs.back());
            double d = dataPoints.size() / double(controlPointsNum - degree);
            for (int j = 1; j <= controlPointsNum - 1 - degree; j++) {
                int i = floor(j * d);
                double alpha = j * d - i;
                knotVector[degree + j] = (1 - alpha) * correspondingUs[i - 1] + alpha * correspondingUs[i];
            }

            return optimization();
        };

        /*
         * Compute nonzero basis functions and their derivatives up to derivativeLimit.
         * N_{i-p,p}(u),...,N_{i,p}(u)
         * Output: two-dimensional array, ders. ders[k][j] is the kth derivative of the function N_{i-p+j,p}, where 0<=k<=n and 0<=j<=p.
         */
        int dersBasisFuns(double u, size_t spanIdx, int derivativeLimit, std::vector<std::vector<double>> &ders) const {
            // check
            if (knotVector.empty()) {
                std::cerr << "Function dersBasisFuns: knotVector is empty!" << std::endl;
                return -1;
            }

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
                left[j] = u - knotVector[spanIdx + 1 - j];
                right[j] = knotVector[spanIdx + j] - u;
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

            return 0;
        }

        /*
         * Determine the knot span index i \in [p,n], whole is [0,n+p+1]
         * u \in [u_{i}, u_{i+1}) , special at u==u_{n+1}
         */
        size_t findSpan(double u) const {
            // assume it's totally p-smooth, which means there is only one knot for each boundary
            // Notes: the size of knot vector is n+p+2
            size_t n = knotVector.size() - 2 - degree;

            // special case
            if (u == knotVector[n + 1])
                return n;

            // binary search
            size_t low = degree;
            size_t high = n + 1;
            size_t mid = (low + high) / 2;
            while (u < knotVector[mid] || u >= knotVector[mid + 1]) {
                if (u < knotVector[mid])
                    high = mid;
                else
                    low = mid;

                mid = (low + high) / 2;
            }

            return mid;
        };

        // shape won't change
        int insertKnot(double u) {
            if (knotVector.empty()) {
                std::cerr << "Function insertKnot: knotVector is empty!" << std::endl;
                return -1;
            }
            if (controlPoints.empty()) {
                std::cerr << "Function insertKnot: controlPoints is empty!" << std::endl;
                return -1;
            }

            size_t spanIdx = findSpan(u);
            std::vector<Eigen::Matrix<double, dim, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, dim,
                    1 >>> Q(controlPoints.size() + 1);

            std::copy(controlPoints.begin(), controlPoints.begin() + spanIdx - degree + 1, Q.begin());
            std::copy(controlPoints.begin() + spanIdx, controlPoints.end(), Q.begin() + spanIdx + 1);
            for (size_t i = spanIdx - degree + 1; i <= spanIdx; i++) {
                double alpha = (u - knotVector[i]) / (knotVector[i + degree] - knotVector[i]);
                Q[i] = alpha * controlPoints[i] + (1 - alpha) * controlPoints[i - 1];
            }

            //update knotVector
            knotVector.insert(knotVector.begin() + spanIdx + 1, u);
            //update controlPoints
            controlPoints = std::move(Q);

            return 0;
        };

        // assume X: ascending sorted
        int refineKnotVect(std::vector<double> &X) {
            if (knotVector.empty()) {
                std::cerr << "Function refineKnotVect: knotVector is empty!" << std::endl;
                return -1;
            }
            if (controlPoints.empty()) {
                std::cerr << "Function refineKnotVect: controlPoints is empty!" << std::endl;
                return -1;
            }

            std::sort(X.begin(), X.end());

            std::vector<Eigen::Matrix<double, dim, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, dim, 1>>> Q(
                    controlPoints.size() + X.size());
            std::vector<double> Ubar(knotVector.size() + X.size());

            size_t n = knotVector.size() - 2 - degree;
            size_t r = X.size() - 1;
            size_t m = knotVector.size() - 1;
            size_t a = findSpan(X.front());
            size_t b = findSpan(X.back());
            b++;

            for (size_t j = 0; j <= a - degree; j++)
                Q[j] = controlPoints[j];
            for (size_t j = b - 1; j <= n; j++)
                Q[j + X.size()] = controlPoints[j];
            for (size_t j = 0; j <= a; j++)
                Ubar[j] = knotVector[j];
            for (size_t j = b + degree; j <= m; j++)
                Ubar[j + X.size()] = knotVector[j];

            int i = b + degree - 1;
            int k = b + degree + r;
            for (int j = r; j >= 0; j--) {
                while (X[j] <= knotVector[i] && i > a) {
                    Q[k - degree - 1] = controlPoints[i - degree - 1];
                    Ubar[k] = knotVector[i];
                    k--;
                    i--;
                }

                Q[k - degree - 1] = Q[k - degree];
                for (int l = 1; l <= degree; l++) {
                    int idx = k - degree + l;
                    double alpha = Ubar[k + l] - X[j];
                    if (std::abs(alpha) == 0.0)
                        Q[idx - 1] = Q[idx];
                    else {
                        alpha /= (Ubar[k + l] - knotVector[i - degree + l]);
                        Q[idx - 1] = alpha * Q[idx - 1] + (1 - alpha) * Q[idx];
                    }
                }
                Ubar[k] = X[j];
                k--;
            }

            //update knotVector
            knotVector = std::move(Ubar);
            //update controlPoints
            controlPoints = std::move(Q);
            return 0;
        };

        // optimize control points w.r.t distance error after knot vector change
        int optimization() {
            /*** check ***/
            if (dataPoints.empty()) {
                std::cerr << "Function optimize: dataPoints is empty!" << std::endl;
                return -1;
            }
            if (dataPoints.size() != derivatives.size() && derivatives.size() != 0) {
                std::cerr
                        << "Function optimization: dataPoints ans derivatives should be same size (NAN for unknow) OR derivatives should be empty!"
                        << std::endl;
                return -1;
            }
            if (knotVector.empty()) {
                std::cerr << "Function optimization: knotVector is empty!" << std::endl;
                return -1;
            }

            int controlPointsNum = knotVector.size() - degree - 1;

            // initialize controlPoints
            controlPoints.resize(controlPointsNum);
            controlPoints.front() = dataPoints.front();
            controlPoints.back() = dataPoints.back();

            // support for derivative lacking
            int derivativesCounter = 0;
            std::vector<int> map(derivatives.size(), -1);
            for (size_t k = 0; k < derivatives.size(); k++) {
                if (!std::isnan(derivatives[k][0])) {
                    map[k] = derivativesCounter++;
                }
            }

            /*** Problem ***/
            Eigen::SparseMatrix<double> N(dataPoints.size(), controlPointsNum);
            Eigen::SparseMatrix<double> M(dataPoints.size(), controlPointsNum);
            std::vector<std::vector<double>> ders;
            for (int k = 0; k < dataPoints.size(); ++k) {
                size_t spanIdx = findSpan(correspondingUs[k]);
                dersBasisFuns(correspondingUs[k], spanIdx, derivativesCounter > 0 ? 1 : 0, ders);

                for (int j = 0; j <= degree; ++j) {
                    if (ders[0][j] != 0)
                        N.insert(k, spanIdx - degree + j) = ders[0][j]; // N_{i-p+j,p}^{0}(\bar{u}_k)

                    if (derivativesCounter > 0) {
                        if (ders[1][j] != 0)
                            M.insert(k, spanIdx - degree + j) = ders[1][j]; // N_{i-p+j,p}^{1}(\bar{u}_k)
                    }
                }
            }

            // formulating B
            std::vector<Eigen::Matrix<double, dim, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, dim, 1>>> B;
            B.assign(controlPointsNum - 2, Eigen::Matrix<double, dim, 1>::Zero());
            for (int l = 1; l <= controlPointsNum - 2; ++l) {
                // for l-th col of N
                for (Eigen::SparseMatrix<double>::InnerIterator it(N, l); it; ++it) {
                    // N_{l,p}(\bar{u}_k), valid for column-major
                    int k = it.row();
                    if (k != 0 && k != dataPoints.size() - 1) {
                        B[l - 1] += it.value() * (dataPoints[k] - N.coeff(k, 0) * dataPoints.front() -
                                                  N.coeff(k, controlPointsNum - 1) * dataPoints.back());
                    }
                }

                if (derivativesCounter > 0) {
                    // for l-th col of M
                    for (Eigen::SparseMatrix<double>::InnerIterator it(M, l); it; ++it) {
                        // N_{l,p}^{1}(\bar{u}_k), valid for column-major
                        int k = it.row();
                        if (map[k] > -1) {
                            B[l - 1] += lambda * it.value() * (derivatives[k] - M.coeff(k, 0) * dataPoints.front() -
                                                               M.coeff(k, controlPointsNum - 1) *
                                                               dataPoints.back());
                        }
                    }
                }
            }

            // formulating M,N
            Eigen::SparseMatrix<double> Nc = N.block(1, 1, dataPoints.size() - 2, controlPointsNum - 2);
            Eigen::SparseMatrix<double> A = Nc.transpose() * Nc;

            if (derivativesCounter > 0) {
                Eigen::SparseMatrix<double> Mc(derivativesCounter, controlPointsNum - 2);
                for (int i = 1; i <= controlPointsNum - 2; ++i) {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(M, i); it; ++it) {// i-th col
                        if (map[it.row()] > -1) {
                            Mc.insert(map[it.row()], it.col() - 1) = it.value();
                        }
                    }
                }
                A += lambda * Mc.transpose() * Mc;
            }

            // solving
            A.makeCompressed();
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(A);
            if (solver.info() != Eigen::Success) {
                std::cerr << "Function optimization: decomposition failed!" << std::endl;
                return -1;
            }
            Eigen::VectorXd b(controlPointsNum - 2);
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < B.size(); ++j) {
                    b[j] = B[j][i];
                }
                Eigen::VectorXd x = solver.solve(b);
                if (solver.info() != Eigen::Success) {
                    std::cerr << "Function optimization: solving failed!" << std::endl;
                    return -1;
                }
                for (int j = 1; j <= controlPointsNum - 2; ++j) {
                    controlPoints[j][i] = x[j - 1];
                }
            }

            return 0;
        };

        /*
         * Evaluate up to k-th derivative at u
         */
        int evaluate(double u, int derivativeLimit,
                     std::vector<Eigen::Matrix<double, dim, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, dim, 1>>> &Ders) const {
            if (controlPoints.empty()) {
                std::cerr << "Function evaluate: controlPoints is empty!" << std::endl;
                return -1;
            }

            Ders.assign(derivativeLimit + 1, Eigen::Matrix<double, dim, 1>::Zero());
            std::vector<std::vector<double>> ders;
            size_t spanIdx = findSpan(u);

            dersBasisFuns(u, spanIdx, derivativeLimit, ders);

            for (size_t i = 0; i < ders.size(); i++) {
                for (size_t j = 0; j < ders[i].size(); ++j) {
                    Ders[i] += ders[i][j] * controlPoints[spanIdx - degree + j];
                }
            }

            return 0;
        }

        inline std::vector<Eigen::Matrix<double, dim, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, dim, 1>>> &
        getCP() {
            return controlPoints;
        };

        inline const std::vector<double> &getCorrespondingUs() {
            return correspondingUs;
        }

        inline const std::vector<double> &getKnotVector() {
            return knotVector;
        }

    protected:
        int degree;

        std::vector<double> knotVector;
        std::vector<Eigen::Matrix<double, dim, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, dim, 1>>> controlPoints;

        // optional
        std::vector<Eigen::Matrix<double, dim, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, dim, 1>>> dataPoints;
        std::vector<double> correspondingUs; //corresponding u for each data point
        std::vector<Eigen::Matrix<double, dim, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, dim, 1>>> derivatives; //corresponding derivatives for each data point
        double lambda;
    };
}

#endif //OPENGV2_BSPLINEREAL_HPP
