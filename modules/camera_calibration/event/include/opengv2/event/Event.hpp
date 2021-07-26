//
// Created by huangkun on 2020/9/9.
//

#ifndef OPENGV2_EVENT_HPP
#define OPENGV2_EVENT_HPP

#include <Eigen/Eigen>
#include <vector>

namespace opengv2 {
    class Event {
        friend std::istream &operator>>(std::istream &is, Event &obj);

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Event(double timeStamp, const Eigen::Ref<const Eigen::Vector2d> &location, bool polarity) :
                timeStamp_(timeStamp), location_(location), polarity_(polarity) {};

        Event() {};

        inline double timeStamp() const noexcept {
            return timeStamp_;
        }

        inline const Eigen::Vector2d &location() const noexcept {
            return location_;
        }

        inline bool polarity() const noexcept {
            return polarity_;
        }

    protected:
        double timeStamp_;
        Eigen::Vector2d location_; // (width, height), (x,y)
        bool polarity_; // true is positive, false is negative
    };

    inline std::istream &operator>>(std::istream &is, Event &obj) {
        is.read((char *) &(obj.timeStamp_), sizeof(obj.timeStamp_));
        is.read((char *) (obj.location_.data()), sizeof(*(obj.location_.data())));
        is.read((char *) (obj.location_.data() + 1), sizeof(*(obj.location_.data() + 1)));
        is.read((char *) &(obj.polarity_), sizeof(obj.polarity_));
        return is;
    }
}

#endif //OPENGV2_EVENT_HPP
