//
// Created by huangkun on 2020/8/20.
//

#ifndef OPENGV2_BSPLINESO3_HPP
#define OPENGV2_BSPLINESO3_HPP

#include <Eigen/StdVector>
#include <Eigen/Eigen>
#include <sophus/so3.hpp>
#include <ceres/ceres.h>

namespace opengv2 {
    class BsplineSO3 {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        explicit BsplineSO3(int p = 4,
                            const std::vector<Sophus::SO3d, Eigen::aligned_allocator<Sophus::SO3d>> &samples = std::vector<Sophus::SO3d, Eigen::aligned_allocator<Sophus::SO3d>>(),
                            int controlPointsNum = -1,
                            const std::vector<double> &correspondingUs = std::vector<double>(),
                            const std::vector<std::vector<Eigen::Vector3d>> &derivativeSamples = std::vector<std::vector<Eigen::Vector3d>>(),
                            double derivativeWeight = 1);

        /*
         * Compute \beta_{k,i} and its derivatives,
         * 0-th derivative: [k-p, k]
         * other: [k-p+1, k]
         */
        void derBasisFuns(double u, size_t spanIdx, int derivativeLimit, std::vector<std::vector<double>> &ders) const;

        /*
         * Evaluate rotation, and angular velocity/acceleration
         */
        void evaluate(double u, int derivativeLimit, Sophus::SO3d &val, std::vector<Eigen::Vector3d> &Ders) const;

        /*
         * u \in [u_{i}, u_{i+1}) , special at u==u_{n+1}
         */
        inline size_t findSpan(double u) const {
            // assume it's totally p-smooth, which means there is only one knot for each boundary
            // Notes: the size of knot vector is n+p+2
            size_t n = knotVector_.size() - 2 - degree_;

            // special case
            if (u == knotVector_[n + 1])
                return n;

            // binary search
            size_t low = degree_;
            size_t high = n + 1;
            size_t mid = (low + high) / 2;
            while (u < knotVector_[mid] || u >= knotVector_[mid + 1]) {
                if (u < knotVector_[mid])
                    high = mid;
                else
                    low = mid;

                mid = (low + high) / 2;
            }

            return mid;
        }

        inline std::vector<Sophus::SO3d, Eigen::aligned_allocator<Sophus::SO3d>> &controlPoints() {
            return controlPoints_;
        };

        inline const std::vector<double> &correspondingUs() const {
            return correspondingUs_;
        }

        inline const std::vector<double> &knotVector() const {
            return knotVector_;
        }

        /*
         * Ceres Residuals, assume p=4
         * TODO: implement analytic Jacobians w.r.t control points in SO3
         */
        struct P4ApproximationError {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            P4ApproximationError(const Sophus::SO3d &sampleInv, std::shared_ptr<std::vector<std::vector<double>>> basis)
                    : sampleInv_(sampleInv), basis_(basis) {}

            template<typename T>
            bool operator()(const T *const cp_0, const T *const cp_1, const T *const cp_2, const T *const cp_3,
                            const T *const cp_4, T *sResiduals) const {
                Eigen::Map<Sophus::SO3<T> const> const cp0(cp_0);
                Eigen::Map<Sophus::SO3<T> const> const cp1(cp_1);
                Eigen::Map<Sophus::SO3<T> const> const cp2(cp_2);
                Eigen::Map<Sophus::SO3<T> const> const cp3(cp_3);
                Eigen::Map<Sophus::SO3<T> const> const cp4(cp_4);
                Eigen::Map<Eigen::Matrix<T, Sophus::SO3d::DoF, 1> > residuals(sResiduals);

                Sophus::SO3<T> X = cp0;
                X *= Sophus::SO3<T>::exp(T(basis_->at(0)[0]) * ((cp0.inverse() * cp1).log())); // j=1
                X *= Sophus::SO3<T>::exp(T(basis_->at(0)[1]) * ((cp1.inverse() * cp2).log())); // j=2
                X *= Sophus::SO3<T>::exp(T(basis_->at(0)[2]) * ((cp2.inverse() * cp3).log())); // j=3
                X *= Sophus::SO3<T>::exp(T(basis_->at(0)[3]) * ((cp3.inverse() * cp4).log())); // j=4

                residuals = (sampleInv_.cast<T>() * X).log(); // Identity tangent = [0 0 0]
                return true;
            }

            static ceres::CostFunction *
            Create(const Sophus::SO3d &sampleInv, std::shared_ptr<std::vector<std::vector<double>>> basis) {
                return (new ceres::AutoDiffCostFunction<P4ApproximationError, Sophus::SO3d::DoF, Sophus::SO3d::num_parameters, Sophus::SO3d::num_parameters, Sophus::SO3d::num_parameters, Sophus::SO3d::num_parameters, Sophus::SO3d::num_parameters>(
                        new P4ApproximationError(sampleInv, basis)));
            }

            Sophus::SO3d sampleInv_;
            std::shared_ptr<std::vector<std::vector<double>>> basis_; // \beta_{k, k-p+j}, j \in [1, p].
        };

        /*
         * Ceres Residuals, assume p=3
         * TODO: implement analytic Jacobians w.r.t control points in SO3
         */
        struct P3ApproximationError {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            P3ApproximationError(const Sophus::SO3d &sampleInv, std::shared_ptr<std::vector<std::vector<double>>> basis)
                    : sampleInv_(sampleInv), basis_(basis) {}

            template<typename T>
            bool operator()(const T *const cp_0, const T *const cp_1, const T *const cp_2, const T *const cp_3,
                            T *sResiduals) const {
                Eigen::Map<Sophus::SO3<T> const> const cp0(cp_0);
                Eigen::Map<Sophus::SO3<T> const> const cp1(cp_1);
                Eigen::Map<Sophus::SO3<T> const> const cp2(cp_2);
                Eigen::Map<Sophus::SO3<T> const> const cp3(cp_3);
                Eigen::Map<Eigen::Matrix<T, Sophus::SO3d::DoF, 1> > residuals(sResiduals);

                Sophus::SO3<T> X = cp0;
                X *= Sophus::SO3<T>::exp(T(basis_->at(0)[0]) * ((cp0.inverse() * cp1).log())); // j=1
                X *= Sophus::SO3<T>::exp(T(basis_->at(0)[1]) * ((cp1.inverse() * cp2).log())); // j=2
                X *= Sophus::SO3<T>::exp(T(basis_->at(0)[2]) * ((cp2.inverse() * cp3).log())); // j=3

                residuals = (sampleInv_.cast<T>() * X).log(); // Identity tangent = [0 0 0]
                return true;
            }

            static ceres::CostFunction *
            Create(const Sophus::SO3d &sampleInv, std::shared_ptr<std::vector<std::vector<double>>> basis) {
                return (new ceres::AutoDiffCostFunction<P3ApproximationError, Sophus::SO3d::DoF, Sophus::SO3d::num_parameters, Sophus::SO3d::num_parameters, Sophus::SO3d::num_parameters, Sophus::SO3d::num_parameters>(
                        new P3ApproximationError(sampleInv, basis)));
            }

            Sophus::SO3d sampleInv_;
            std::shared_ptr<std::vector<std::vector<double>>> basis_; // \beta_{k, k-p+j}, j \in [1, p].
        };

    protected:
        void knotSpacing(int controlPointsNum);

        /*
         * Initial guess in R^d space
         */
        void initialGuess();

        /*
         * Optimizing in Lie Group using ceres
         */
        void optimizeCP();

        /*
         * Compute R^d B-spline nonzero basis functions and their derivatives up to derivativeLimit.
         * N_{i-p,p}(u),...,N_{i,p}(u)
         * Output: two-dimensional array, ders. ders[k][j] is the kth derivative of the function N_{i-p+j,p}, where 0<=k<=n and 0<=j<=p.
         */
        void
        basis(double u, size_t spanIdx, int derivativeLimit, int degree, std::vector<std::vector<double>> &ders) const;

        int degree_; // polynomial degree p = order + 1 (p=4 is cubic spline)

        std::vector<double> knotVector_;
        std::vector<Sophus::SO3d, Eigen::aligned_allocator<Sophus::SO3d>> controlPoints_;

        // optional
        std::vector<Sophus::SO3d, Eigen::aligned_allocator<Sophus::SO3d>> samples_;
        std::vector<double> correspondingUs_; //corresponding u for each data point

        // TODO: add angular velocity/acceleration constraint
        std::vector<std::vector<Eigen::Vector3d>> derivativeSamples_; //corresponding derivatives for each data point
        double derivativeWeight_;
    };

    class LocalParameterizationSO3 : public ceres::LocalParameterization {
    public:
        // SO3 plus operation for Ceres
        //
        //  T * exp(x)
        //
        bool Plus(double const *T_raw, double const *delta_raw,
                  double *T_plus_delta_raw) const override {
            Eigen::Map<Sophus::SO3d const> const T(T_raw);
            Eigen::Map<Eigen::Matrix<double, Sophus::SO3d::DoF, 1> const> const delta(delta_raw);
            Eigen::Map<Sophus::SO3d> T_plus_delta(T_plus_delta_raw);
            T_plus_delta = T * Sophus::SO3d::exp(delta);
            return true;
        }

        // Jacobian of SO3 plus operation for Ceres
        //
        // Dx T * exp(x)  with  x=0
        //
        bool ComputeJacobian(double const *T_raw,
                             double *jacobian_raw) const override {
            Eigen::Map<Sophus::SO3d const> T(T_raw);
            Eigen::Map<Eigen::Matrix<double, Sophus::SO3d::num_parameters, Sophus::SO3d::DoF, Eigen::RowMajor>> jacobian(
                    jacobian_raw);
            jacobian = T.Dx_this_mul_exp_x_at_0();
            return true;
        }

        int GlobalSize() const override { return Sophus::SO3d::num_parameters; }

        int LocalSize() const override { return Sophus::SO3d::DoF; }
    };
}

#endif //OPENGV2_BSPLINESO3_HPP
