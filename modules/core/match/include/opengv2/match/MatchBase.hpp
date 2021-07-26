//
// Created by huangkun on 2020/2/15.
//

#ifndef OPENGV2_MATCHBASE_HPP
#define OPENGV2_MATCHBASE_HPP

#include <memory>

#include <opengv2/feature/FeatureIdentifier.hpp>

namespace opengv2 {
    enum MatchType {
        MATCH2D2D,
        MATCH2D3D,
        MATCH3D3D
    };

    class MatchBase {
    public:
        typedef std::shared_ptr<MatchBase> Ptr;

        MatchBase(MatchType matchType, double distance) : matchType_(matchType), distance_(distance) {};

        virtual ~MatchBase() = default;

        inline double distance() const noexcept {
            return distance_;
        }

        inline MatchType matchType() const noexcept {
            return matchType_;
        }

    protected:
        MatchType matchType_;
        double distance_;
    };

    class Match2D2D : public MatchBase {
    public:
        Match2D2D(const FeatureIdentifier &fi1, const FeatureIdentifier &fi2, double distance)
                : MatchBase(MATCH2D2D, distance), fi1_(fi1), fi2_(fi2) {};

        inline bool operator==(const Match2D2D &other) const noexcept {
            return fi1_ == other.fi1_ && fi2_ == other.fi2_;
        }

        inline const FeatureIdentifier &fi1() const noexcept {
            return fi1_;
        }

        inline const FeatureIdentifier &fi2() const noexcept {
            return fi2_;
        }

    protected:
        FeatureIdentifier fi1_;
        FeatureIdentifier fi2_;
    };

    class Match2D3D : public MatchBase {
    public:
        Match2D3D(FeatureIdentifier &fi1, int lmId2, double distance) : MatchBase(MATCH2D3D, distance), fi1_(fi1),
                                                                        lmId2_(lmId2) {};

        inline bool operator==(const Match2D3D &other) const noexcept {
            return fi1_ == other.fi1_ && lmId2_ == other.lmId2_;
        }

        inline const FeatureIdentifier &fi1() const noexcept {
            return fi1_;
        }

        inline int lmId2() const noexcept {
            return lmId2_;
        }

    protected:
        FeatureIdentifier fi1_;
        int lmId2_;
    };

    class Match3D3D : public MatchBase {
    public:
        Match3D3D(int lmId1, int lmId2, double distance) : MatchBase(MATCH3D3D, distance), lmId1_(lmId1),
                                                           lmId2_(lmId2) {};

        inline bool operator==(const Match3D3D &other) const noexcept {
            return lmId1_ == other.lmId1_ && lmId2_ == other.lmId2_;
        }

        inline int lmId1() const noexcept {
            return lmId1_;
        }

        inline int lmId2() const noexcept {
            return lmId2_;
        }

    protected:
        int lmId1_;
        int lmId2_;
    };
}

#endif //OPENGV2_MATCHBASE_HPP
