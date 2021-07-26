/*
 * Copyright (c) 2021. Kun Huang.
 * This Source Code Form is subject to the terms of the Apache License, v. 2.0.
 */

//
// Created by huangkun on 2021/7/9.
//

#ifndef OPENGV2_UTILITY_HPP
#define OPENGV2_UTILITY_HPP

#include <functional>
#include <numeric>
#include <unordered_map>

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <nanoflann.hpp>
//#include <cereal/cereal.hpp>

namespace opengv2 {
    template<typename EigenMatrixType>
    using vectorofEigenMatrix = std::vector<EigenMatrixType, Eigen::aligned_allocator<EigenMatrixType>>;

    template<typename T>
    struct EigenMatrixCompare {
        inline bool operator()(const T &lhs, const T &rhs) const {
            return lhs.norm() < rhs.norm();
        }
    };

    /**
     * @brief Hash function for Eigen matrix and vector.
     * The code is from `hash_combine` function of the Boost library. See
     * http://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine .
     */
    template<typename T>
    struct EigenMatrixHash : std::unary_function<T, size_t> {
        std::size_t operator()(T const &matrix) const {
            // Note that it is oblivious to the storage order of Eigen matrix (column- or
            // row-major). It will give you the same hash value for two different matrices if they
            // are the transpose of each other in different storage order.
            size_t seed = 0;
            for (size_t i = 0; i < matrix.size(); ++i) {
                auto elem = *(matrix.data() + i);
                seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    /**
     * @brief Generic functor base for use with the Eigen-nonlinear optimization
     * toolbox. Please refer to the Eigen-documentation for further information.
     */
    template<typename Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
    struct EigenOptimizationFunctor {
        enum {
            InputsAtCompileTime = NX,
            ValuesAtCompileTime = NY
        };
        typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
        typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
        typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

        const int m_inputs, m_values;

        EigenOptimizationFunctor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}

        EigenOptimizationFunctor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

        int inputs() const { return m_inputs; }

        int values() const { return m_values; }

        // you should define that in the subclass :
        // void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
    };

    struct pair_hash {
        template<class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &pair) const {
            return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
        }
    };

    /**
     * @brief Fit normal distribution.
     * @tparam T
     * @param v
     * @param mean
     * @param stdev
     */
    template<class T>
    inline void fitNormal(const std::vector<T> &v, T &mean, T &stdev) {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        mean = sum / v.size();

        std::vector<T> diff(v.size());
        std::transform(v.begin(), v.end(), diff.begin(), [mean](T x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        stdev = std::sqrt(sq_sum / (v.size() - 1));
    }
}

namespace nanoflann {
    template<typename T>
    struct SO3DataSetAdaptor {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        explicit SO3DataSetAdaptor(const opengv2::vectorofEigenMatrix<Eigen::Quaternion<T>> &Q_set) : QSet(Q_set) {}

        const opengv2::vectorofEigenMatrix<Eigen::Quaternion<T>> &QSet;

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const { return QSet.size(); }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
            return QSet[idx].coeffs()[dim];
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
        //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template<class BBOX>
        bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
    };

    template<typename num_t>
    using SO3_KDTree = KDTreeSingleIndexAdaptor<SO3_Adaptor<num_t, SO3DataSetAdaptor<num_t>>, SO3DataSetAdaptor<num_t>, 4>;
}

/*namespace cereal {
    template<class Archive, class Derived>
    inline
    typename std::enable_if<
            traits::is_output_serializable < BinaryData < typename Derived::Scalar>, Archive>::value, void>

    ::type
    save(Archive &ar, Eigen::PlainObjectBase<Derived> const &m) {
        typedef Eigen::PlainObjectBase<Derived> ArrT;
        if (ArrT::RowsAtCompileTime == Eigen::Dynamic) ar(m.rows());
        if (ArrT::ColsAtCompileTime == Eigen::Dynamic) ar(m.cols());
        ar(binary_data(m.data(), m.size() * sizeof(typename Derived::Scalar)));
    }

    template<class Archive, class Derived>
    inline
    typename std::enable_if<
            traits::is_input_serializable < BinaryData < typename Derived::Scalar>, Archive>::value, void>

    ::type
    load(Archive &ar, Eigen::PlainObjectBase<Derived> &m) {
        typedef Eigen::PlainObjectBase<Derived> ArrT;
        Eigen::Index rows = ArrT::RowsAtCompileTime, cols = ArrT::ColsAtCompileTime;
        if (rows == Eigen::Dynamic) ar(rows);
        if (cols == Eigen::Dynamic) ar(cols);
        m.resize(rows, cols);
        ar(binary_data(m.data(), static_cast<std::size_t>(rows * cols * sizeof(typename Derived::Scalar))));
    }

    template<class Archive, class Scalar, int Options>
    inline void serialize(Archive &ar, ::Eigen::Quaternion<Scalar, Options> &quat) {
        ar(make_nvp("w", quat.w()), make_nvp("x", quat.x()), make_nvp("y", quat.y()), make_nvp("z", quat.z()));
    }
}*/

#endif //OPENGV2_UTILITY_HPP
