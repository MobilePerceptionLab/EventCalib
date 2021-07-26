//
// Created by huangkun on 2020/9/15.
//

#ifndef OPENGV2_EVENTSTREAM_HPP
#define OPENGV2_EVENTSTREAM_HPP

#include <iterator>
#include <memory>
#include <fstream>

#include <opengv2/event/Event.hpp>
#include <opengv2/sensor/CameraBase.hpp>

namespace opengv2 {
    /*
     *  A single-pass input stream
     */
    class EventStream {
    public:
        explicit EventStream(const std::string &binFilePath);

        ~EventStream();

        inline void close() {
            if (is_.is_open())
                is_.close();
        }

        inline bool isEnd() const noexcept {
            if (is_.is_open()) {
                return istreamIterator_ == std::istream_iterator<Event>();
            } else {
                return true;
            }
        }

        inline std::istream_iterator<Event> &iterator() noexcept {
            return istreamIterator_;
        }

        static void txt2bin(const std::string &txtFilePath, double timeMagnitude = 1e-6,
                            long long timeBase_in = std::numeric_limits<long long>::min(),
                            long long endTime_in = std::numeric_limits<long long>::min());

    protected:
        std::ifstream is_;
        std::istream_iterator<Event> istreamIterator_;

        // TODO: real time event stream
    };
}

#endif //OPENGV2_EVENTSTREAM_HPP
