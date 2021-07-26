//
// Created by huangkun on 2020/9/28.
//

#include <cv_calib.hpp>
#include <circlesgrid.hpp>

bool cv::findCirclesGrid(const std::vector<Point2f> &points_, Size patternSize, OutputArray _centers, int flags,
                         const CirclesGridFinderParameters &parameters_) {
    CirclesGridFinderParameters parameters = parameters_; // parameters.gridType is amended below

    bool isAsymmetricGrid = (flags & CALIB_CB_ASYMMETRIC_GRID) ? true : false;
    bool isSymmetricGrid = (flags & CALIB_CB_SYMMETRIC_GRID) ? true : false;
    CV_Assert(isAsymmetricGrid ^ isSymmetricGrid);

    std::vector<Point2f> centers;
    std::vector<Point2f> points(points_);

    if (flags & CALIB_CB_ASYMMETRIC_GRID)
        parameters.gridType = CirclesGridFinderParameters::ASYMMETRIC_GRID;
    if (flags & CALIB_CB_SYMMETRIC_GRID)
        parameters.gridType = CirclesGridFinderParameters::SYMMETRIC_GRID;

    if (flags & CALIB_CB_CLUSTERING) {
        // uses a special algorithm for grid detection. It is more robust to perspective distortions
        // but much more sensitive to background clutter.
        CirclesGridClusterFinder circlesGridClusterFinder(parameters);
        circlesGridClusterFinder.findGrid(points, patternSize, centers);
        Mat(centers).copyTo(_centers);
        return !centers.empty();
    }

    const int attempts = 2;
    const size_t minHomographyPoints = 4;
    Mat H;
    for (int i = 0; i < attempts; i++) {
        centers.clear();
        CirclesGridFinder boxFinder(patternSize, points, parameters);
        bool isFound = false;
        try {
            isFound = boxFinder.findHoles();
        }
        catch (const cv::Exception &) {

        }

        if (isFound) {
            switch (parameters.gridType) {
                case CirclesGridFinderParameters::SYMMETRIC_GRID:
                    boxFinder.getHoles(centers);
                    break;
                case CirclesGridFinderParameters::ASYMMETRIC_GRID:
                    boxFinder.getAsymmetricHoles(centers);
                    break;
                default:
                    CV_Error(Error::StsBadArg, "Unknown pattern type");
            }

            // add by huangkun
            if (centers.empty()) {
                /*std::cout << "debug" << std::endl;
                FileStorage fout("/home/huangkun/debug.yml", FileStorage::WRITE);
                fout << "candidateCenters" << points_;
                fout << "patternSize" << patternSize;
                fout << "flags" << flags;
                fout.release();*/
                return false;
            }

            if (i != 0) {
                Mat orgPointsMat;
                transform(centers, orgPointsMat, H.inv());
                convertPointsFromHomogeneous(orgPointsMat, centers);
            }
            Mat(centers).copyTo(_centers);
            return true;
        }

        boxFinder.getHoles(centers);
        if (i != attempts - 1) {
            if (centers.size() < minHomographyPoints)
                break;
            H = CirclesGridFinder::rectifyGrid(boxFinder.getDetectedGridSize(), centers, points, points);
        }
    }
    Mat(centers).copyTo(_centers);
    return false;
}