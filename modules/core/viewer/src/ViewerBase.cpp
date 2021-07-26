//
// Created by huangkun on 2020/1/9.
//

#include <numeric>

#include <opengv2/viewer/ViewerBase.hpp>

opengv2::ViewerBase::ViewerBase(const std::string &strSettingPath, MapBase::Ptr map, TrackingBase::Ptr tracking)
        : map_(std::move(map)), tracking_(std::move(tracking)), GT_Tws_(nullptr) {
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Camera.fps"];
    if (fps < 1)
        fps = 30;
    T_ = 1e3 / fps;

    viewpointX_ = fSettings["Viewer.ViewpointX"];
    viewpointY_ = fSettings["Viewer.ViewpointY"];
    viewpointZ_ = fSettings["Viewer.ViewpointZ"];
    viewpointF_ = fSettings["Viewer.ViewpointF"];

    cameraSize_ = fSettings["Viewer.CameraSize"];
    cameraLineWidth_ = fSettings["Viewer.CameraLineWidth"];
    keyFrameSize_ = fSettings["Viewer.KeyFrameSize"];
    keyFrameLineWidth_ = fSettings["Viewer.KeyFrameLineWidth"];
    graphLineWidth_ = fSettings["Viewer.GraphLineWidth"];

    stopped_ = false;
    stopRequested_ = false;
    finishRequested_ = false;
    finished_ = false;

    showLocal_ = false;
    localRange_ = 100;

    std::vector<double> v;
    cv::FileNode data = fSettings["Viewer.Facing"];
    if (data.empty()) {
        Transform_ << 1, 0, 0, 0,
                0, 0, 1, 0,
                0, -1, 0, 0,
                0, 0, 0, 1;
    } else {
        for (cv::FileNodeIterator itData = data.begin(); itData != data.end(); ++itData) {
            v.push_back(*itData);
        }
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R(v.data());
        Transform_ = Eigen::Matrix4d::Identity();
        Transform_.block<3, 3>(0, 0) = R;
    }

    update_ = true;
}

void opengv2::ViewerBase::run() {
    pangolin::CreateWindowAndBind("Map Viewer", 1024, 768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
    pangolin::Var<bool> menuDrawCamera("menu.Draw Camera", true, true);
    pangolin::Var<bool> menuDrawConnection("menu.Draw Connection", true, true);
    pangolin::Var<bool> menuShowGT("menu.Show Groundtruth", true, true);
    pangolin::Var<bool> menuDrawMatches("menu.Draw Matches", true, true);
    pangolin::Var<bool> menuDrawComparison("menu.Draw Comparison", true, true);
    pangolin::Var<bool> menuDrawLocal("menu.Only Local", false, true);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, viewpointF_, viewpointF_, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(viewpointX_, viewpointY_, viewpointZ_, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix glTwc;
    glTwc.SetIdentity();

    cv::namedWindow("Current Frame");

    bool bFollow = true;

    while (true) {
        while (!checkSystemState()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        updateMutex_.lock_shared();
        if (update_) {
            updateMutex_.unlock_shared();
            updateData();
            keyframes_ = map_->copyKeyframes();
            landmarks_ = map_->copyLandmarks();
            std::scoped_lock lock(updateMutex_);
            update_ = false;
        } else {
            updateMutex_.unlock_shared();
        }

        Bodyframe::Ptr bf = keyframes_.crbegin()->second;
        if (bf == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        getCurrentOpenGLMatrix(bf, glTwc);

        if (menuFollowCamera && bFollow) {
            s_cam.Follow(glTwc);
        } else if (menuFollowCamera && !bFollow) {
            s_cam.SetModelViewMatrix(
                    pangolin::ModelViewLookAt(viewpointX_, viewpointY_, viewpointZ_, 0, 0, 0, 0.0, -1.0, 0.0));
            s_cam.Follow(glTwc);
            bFollow = true;
        } else if (!menuFollowCamera && bFollow) {
            bFollow = false;
        }

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        drawCurrentFrame(glTwc);
        showLocal_ = menuDrawLocal;
        if (menuShowKeyFrames)
            drawKeyFrames(menuDrawCamera, menuDrawConnection);
        if (menuShowPoints)
            drawLandmarks();
        if (menuShowGT)
            drawGT(bf);
        if (menuDrawComparison)
            drawComparison();

        pangolin::FinishFrame();
        cv::Mat im = getCurrentImages(bf, menuDrawMatches);
        if (!im.empty()) {
            cv::imshow("Current Frame", im);
            cv::waitKey(int(T_));
        }

        if (stop()) {
            while (isStopped()) {
                std::this_thread::sleep_for(std::chrono::microseconds(3000));
            }
        }

        if (checkFinish())
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    setFinish();
}

void opengv2::ViewerBase::getCurrentOpenGLMatrix(Bodyframe::Ptr bf, pangolin::OpenGlMatrix &glTwc) {
    // If the storage order is not specified, then Eigen defaults to storing the entry in column-major
    // opencv is row-major
    if (bf != nullptr) {
        Eigen::Matrix4d Twb = bf->Twb() * Transform_;
        glTwc = pangolin::OpenGlMatrix(Twb);
    } else {
        glTwc.SetIdentity();
    }
}

void opengv2::ViewerBase::drawCurrentFrame(pangolin::OpenGlMatrix &glTwc) const noexcept {
    const float &w = cameraSize_;
    const float h = w * 0.75f;
    const float z = w * 0.6f;

    glPushMatrix();

#ifdef HAVE_GLES
    glMultMatrixf(glTwc.m);
#else
    glMultMatrixd(glTwc.m);
#endif

    glLineWidth(cameraLineWidth_);
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);

    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();
}

void opengv2::ViewerBase::drawKeyFrames(bool drawCamera,
                                        bool drawConnection) {
    if (drawCamera) {
        const float &w = keyFrameSize_;
        const float h = w * 0.75f;
        const float z = w * 0.6f;

        auto beginItr = keyframes_.end();
        if (showLocal_ && keyframes_.size() > localRange_)
            std::advance(beginItr, -localRange_);
        else
            beginItr = keyframes_.begin();
        for (; beginItr != keyframes_.end(); beginItr++) {
            Bodyframe::Ptr pKF = beginItr->second;
            Eigen::Matrix4d glTwb = pKF->Twb() * Transform_;

            glPushMatrix();

            glMultMatrixd((GLdouble *) glTwb.data());

            glLineWidth(keyFrameLineWidth_);
            glColor3f(0.0f, 0.0f, 1.0f);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(w, h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(w, -h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(-w, -h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(-w, h, z);

            glVertex3f(w, h, z);
            glVertex3f(w, -h, z);

            glVertex3f(-w, h, z);
            glVertex3f(-w, -h, z);

            glVertex3f(-w, h, z);
            glVertex3f(w, h, z);

            glVertex3f(-w, -h, z);
            glVertex3f(w, -h, z);
            glEnd();

            glPopMatrix();
        }
    }

    if (drawConnection) {
        // Draw connection
        glLineWidth(graphLineWidth_);
        glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
        glBegin(GL_LINES);

        auto it = keyframes_.end();
        if (showLocal_ && keyframes_.size() > localRange_)
            std::advance(it, -localRange_);
        else
            it = keyframes_.begin();
        for (auto preIt = it++; it != keyframes_.end(); preIt = it++) {
            Eigen::Vector3d Ow = it->second->twb();
            Eigen::Vector3d Owp = preIt->second->twb();

            glVertex3d(Ow(0), Ow(1), Ow(2));
            glVertex3d(Owp(0), Owp(1), Owp(2));
        }

        glEnd();
    }
}

bool opengv2::ViewerBase::stop() {
    std::unique_lock<std::mutex> lock(mutexStop_);
    std::unique_lock<std::mutex> lock2(mutexFinish_);

    if (finishRequested_)
        return false;
    else if (stopRequested_) {
        stopped_ = true;
        stopRequested_ = false;
        return true;
    }

    return false;
}

bool opengv2::ViewerBase::isStopped() {
    std::unique_lock<std::mutex> lock(mutexStop_);
    return stopped_;
}

bool opengv2::ViewerBase::checkFinish() {
    std::unique_lock<std::mutex> lock(mutexFinish_);
    return finishRequested_;
}

void opengv2::ViewerBase::setFinish() {
    std::unique_lock<std::mutex> lock(mutexFinish_);
    finished_ = true;
}

void opengv2::ViewerBase::requestFinish() {
    std::unique_lock<std::mutex> lock(mutexFinish_);
    finishRequested_ = true;
}

void opengv2::ViewerBase::requestStop() {
    std::unique_lock<std::mutex> lock(mutexStop_);
    if (!stopped_)
        stopRequested_ = true;
}

bool opengv2::ViewerBase::isFinished() {
    std::unique_lock<std::mutex> lock(mutexFinish_);
    return finished_;
}

void opengv2::ViewerBase::recovery() {
    std::unique_lock<std::mutex> lock(mutexStop_);
    stopped_ = false;
}

bool opengv2::ViewerBase::checkSystemState() {
    if (tracking_ != nullptr) {
        std::shared_lock lock(tracking_->stateMutex);
        return tracking_->state == TrackingState::OK;
    } else {
        return true;
    }
}

void opengv2::ViewerBase::align(const std::vector<Eigen::Vector3d> &model, const std::vector<Eigen::Vector3d> &data,
                                Eigen::Quaterniond &Qdm, Eigen::Vector3d &tdm) {
    if (model.size() != data.size())
        throw std::logic_error("Viewer: the alignment vector size should the same.");

    Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    Eigen::Vector3d model_mean = std::accumulate(model.begin(), model.end(), zero) / model.size();
    Eigen::Vector3d data_mean = std::accumulate(data.begin(), data.end(), zero) / data.size();

    Eigen::Matrix<double, 3, Eigen::Dynamic> model_zerocentered(3, model.size()), data_zerocentered(3, data.size());
    for (int i = 0; i < model.size(); ++i) {
        model_zerocentered.col(i) = model[i] - model_mean;
        data_zerocentered.col(i) = data[i] - data_mean;
    }
    Eigen::Matrix3d W = (model_zerocentered * data_zerocentered.transpose()).transpose();
    Eigen::JacobiSVD svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0)
        S(2, 2) = -1;
    Qdm = svd.matrixU() * S * svd.matrixV().transpose();
    Qdm.normalize();
    tdm = data_mean - Qdm * model_mean;
}

void opengv2::ViewerBase::drawGT(Bodyframe::Ptr bf) {
    if (keyframes_.empty() || GT_Tws_ == nullptr) {
        return;
    }

    for (int i = 0; i < bf->size(); ++i) {
        if (!GT_Tws_->at(i).empty()) {
            Bodyframe::Ptr bf0 = keyframes_.cbegin()->second;
            auto beginIt = GT_Tws_->at(i).lower_bound(bf0->timeStamp());
            auto curIt = GT_Tws_->at(i).upper_bound(bf->timeStamp());
            auto curF = keyframes_.find(bf->timeStamp());
            bool gtLonger = std::distance(beginIt, curIt) > std::distance(keyframes_.begin(), curF);
            std::vector<Eigen::Vector3d> gt/*model*/, data;
            Eigen::Vector3d tbs = -(Bodyframe::unitQsb(i).conjugate() * Bodyframe::tsb(i));
            if (gtLonger) {
                for (auto itr = keyframes_.begin(); itr != curF; itr++) {
                    auto n1 = GT_Tws_->at(i).lower_bound(itr->first);
                    if (n1 == GT_Tws_->at(i).end()) {
                        n1--;
                    } else if (n1 != GT_Tws_->at(i).begin()) {
                        auto n2 = n1--;
                        n1 = std::abs(itr->first - n1->first) > std::abs(itr->first - n2->first) ? n2 : n1;
                    }

                    auto itr1 = keyframes_.lower_bound(n1->first);
                    if (itr1 == keyframes_.end()) {
                        itr1--;
                    } else if (itr1 != keyframes_.begin()) {
                        auto itr2 = itr1--;
                        itr1 = std::abs(n1->first - itr1->first) > std::abs(n1->first - itr2->first) ? itr2 : itr1;
                    }

                    if (itr1->first != itr->first || std::abs(n1->first - itr->first) > 3e-2)
                        continue;

                    data.emplace_back(itr->second->unitQwb() * tbs + itr->second->twb());
                    gt.emplace_back(n1->second.tail<3>());
                }
            } else {
                for (auto itr = beginIt; itr != curIt; itr++) {
                    auto n1 = keyframes_.lower_bound(itr->first);
                    if (n1 == keyframes_.end()) {
                        n1--;
                    } else if (n1 != keyframes_.begin()) {
                        auto n2 = n1--;
                        n1 = std::abs(itr->first - n1->first) > std::abs(itr->first - n2->first) ? n2 : n1;
                    }

                    auto itr1 = GT_Tws_->at(i).lower_bound(n1->first);
                    if (itr1 == GT_Tws_->at(i).end()) {
                        itr1--;
                    } else if (itr1 != GT_Tws_->at(i).begin()) {
                        auto itr2 = itr1--;
                        itr1 = std::abs(n1->first - itr1->first) > std::abs(n1->first - itr2->first) ? itr2 : itr1;
                    }

                    if (itr1->first != itr->first || std::abs(n1->first - itr->first) > 3e-2)
                        continue;

                    gt.emplace_back(itr->second.tail<3>());
                    data.emplace_back(n1->second->unitQwb() * tbs + n1->second->twb());
                }
            }
            Eigen::Quaterniond Q;
            Eigen::Vector3d t;
            align(gt, data, Q, t);

            auto beginItr = keyframes_.end();
            if (showLocal_ && keyframes_.size() > localRange_)
                std::advance(beginItr, -localRange_);
            else
                beginItr = keyframes_.begin();
            beginIt = GT_Tws_->at(i).lower_bound(beginItr->first);
            std::vector<Eigen::Vector3d> GTs_twb;
            for (auto it = beginIt; it != curIt; it++) {
                const Eigen::Matrix<double, 7, 1> &GT_Tws = it->second;
                Eigen::Map<const Eigen::Quaterniond> GT_Qws(GT_Tws.data());
                Eigen::Map<const Eigen::Vector3d> GT_tws(GT_Tws.data() + 4);
                Eigen::Vector3d GT_twb = GT_Qws * Bodyframe::tsb(i) + GT_tws;
                GTs_twb.emplace_back(Q * GT_twb + t);
            }

            if (GTs_twb.size() > 2) {
                glLineWidth(graphLineWidth_);
                glColor4f(1.0f, 0.0f, 0.0f, 0.6f);
                glBegin(GL_LINES);
                for (int j = 0; j < GTs_twb.size() - 1; ++j) {
                    glVertex3d(GTs_twb[j](0), GTs_twb[j](1), GTs_twb[j](2));
                    glVertex3d(GTs_twb[j + 1](0), GTs_twb[j + 1](1), GTs_twb[j + 1](2));
                }
                glEnd();
            }
        }
    }
}

void opengv2::ViewerBase::drawComparison() {
    // GT: red, cur(FSBA): green, {blue, cyan, black}
    int color[4][3] = {{0, 0, 1}, // blue : CBA
                       {1, 0, 1}, // magenta: CBA+relative
                       {0, 1, 1}, // cyan : CBA + constraint
                       {0, 0, 0}};// black : SSBA + constraint
    int i = 0;
    if (savedTrajectory_.size() > 4) {// TODO: add more color
        return;
    }
    for (auto &trj: savedTrajectory_) {
        glLineWidth(graphLineWidth_);
        glColor4f(color[i][0], color[i][1], color[i][2], 0.6f);
        glBegin(GL_LINES);

        auto preIt = trj.begin();
        auto it = trj.begin();
        for (it++; it != trj.end(); it++, preIt++) {
            Eigen::Vector3d &Ow = *it;
            Eigen::Vector3d &Owp = *preIt;
            glVertex3d(Ow(0), Ow(1), Ow(2));
            glVertex3d(Owp(0), Owp(1), Owp(2));
        }

        glEnd();
        i++;
    }
}

void opengv2::ViewerBase::saveTrajectory() {
    savedTrajectory_.emplace_back();
    map_->keyframeLockShared();
    for (auto &it : map_->keyframes()) {
        auto bf = it.second;
        Eigen::Vector3d Ow = bf->twb();
        savedTrajectory_.back().push_back(Ow);
    }
    map_->keyframeUnlockShared();
}