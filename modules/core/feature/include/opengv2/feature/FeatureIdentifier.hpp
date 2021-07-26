//
// Created by huangkun on 2020/1/1.
//

#ifndef OPENGV2_FEATUREIDENTIFIER_HPP
#define OPENGV2_FEATUREIDENTIFIER_HPP

#include <hash_set>
#include <memory>

namespace opengv2 {
    class FeatureBase;

    class FeatureIdentifier {
    public:
        FeatureIdentifier(double timestamp, int frameId, int featureId, const std::shared_ptr<FeatureBase> &feature) :
                timestamp_(timestamp), frameId_(frameId), featureId_(featureId), feature_(feature) {}

        inline bool operator==(const FeatureIdentifier &other) const noexcept {
            return timestamp_ == other.timestamp_ && featureId_ == other.featureId_ &&
                   frameId_ == other.frameId_;
        }

        inline bool operator<(const FeatureIdentifier &other) const noexcept {
            return timestamp_ < other.timestamp_;
        }

        inline double timestamp() const noexcept {
            return timestamp_;
        }

        inline int frameId() const noexcept {
            return frameId_;
        }

        inline int featureId() const noexcept {
            return featureId_;
        }

        inline std::shared_ptr<FeatureBase> feature() const noexcept {
            return feature_.lock();
        }

        // TODO: is this hash valid?
        struct FeatureIdentifierHash {
            size_t operator()(const FeatureIdentifier &v) const {
                std::hash<int> hasher;
                std::hash<double> hasher1;
                size_t seed = 0;
                seed ^= hasher1(v.timestamp()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                seed ^= hasher(v.frameId()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                seed ^= hasher(v.featureId()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

                return seed;
            }
        };

    protected:
        double timestamp_;
        int frameId_;
        int featureId_;

        /**
         * Another use for std::weak_ptr is to break reference cycles formed by objects managed by std::shared_ptr.
         * If such cycle is orphaned (i,e. there are no outside shared pointers into the cycle),
         * the shared_ptr reference counts cannot reach zero and the memory is leaked.
         * To prevent this, one of the pointers in the cycle can be made weak.
         */
        std::weak_ptr<FeatureBase> feature_; // for fast access, non-owning
    };
}

#endif //OPENGV2_FEATUREIDENTIFIER_HPP
