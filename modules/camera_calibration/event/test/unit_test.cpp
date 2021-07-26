//
// Created by huangkun on 2020/9/17.
//

#include <iostream>

#include <opengv2/event/EventStream.hpp>

using namespace std;
using namespace opengv2;

int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << endl
             << "Usage: ./unit_test_event binFilePath"
             << endl;
        return 1;
    }

    EventStream es((string(argv[1])));

    cout.precision(8);
    for (int i = 0; i < 100; ++i) {
        cout << es.iterator()->timeStamp() << " " << es.iterator()->location().transpose() << " "
             << es.iterator()->polarity() << endl;
        es.iterator()++;
    }

    return 0;
}