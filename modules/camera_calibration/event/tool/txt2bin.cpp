//
// Created by huangkun on 2020/9/18.
//

#include <iostream>

#include <opengv2/event/EventStream.hpp>

using namespace std;
using namespace opengv2;

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << endl
             << "Usage: ./txt2bin txtFilePath timeMagnitude [baseTime] [endTime]"
             << endl;
        return 1;
    }

    double timeMagnitude = std::stod(argv[2]);
    long long baseTime = std::numeric_limits<long long>::min(), endTime = std::numeric_limits<long long>::min();
    if (argc > 3)
        baseTime = std::stoll(argv[3]);
    if (argc > 4)
        endTime = std::stoll(argv[4]);

    EventStream::txt2bin(string(argv[1]), timeMagnitude, baseTime, endTime);

    return 0;
}