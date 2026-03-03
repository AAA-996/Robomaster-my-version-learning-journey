#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int Hmin = 5, Hmax = 20;
int Smin = 100, Smax = 255;
int Vmin = 80, Vmax = 255;

const double minArea = 300.0;
const int morphK = 7;
const bool drawBoundingBox = true;

static cv::Mat makeKernel(int k) {
    k = std::max(1, k);
    return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
}

void onTrackbarChange(int, void*) {}

int main(int argc, char** argv) {
    std::string videoPath = (argc >= 2) ? argv[1] : "/home/noah/work/assets/origin.mp4";
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << videoPath << std::endl;
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int waitTime = fps > 0 ? static_cast<int>(1000.0 / fps) : 33;

    cv::namedWindow("Threshold_debug", cv::WINDOW_NORMAL);
    cv::resizeWindow("Threshold_debug", 1500, 600);
    cv::moveWindow("Threshold_debug", 50, 50);

    cv::createTrackbar("H Min", "Threshold_debug", &Hmin, 179, onTrackbarChange);
    cv::createTrackbar("H Max", "Threshold_debug", &Hmax, 179, onTrackbarChange);
    cv::createTrackbar("S Min", "Threshold_debug", &Smin, 255, onTrackbarChange);
    cv::createTrackbar("S Max", "Threshold_debug", &Smax, 255, onTrackbarChange);
    cv::createTrackbar("V Min", "Threshold_debug", &Vmin, 255, onTrackbarChange);
    cv::createTrackbar("V Max", "Threshold_debug", &Vmax, 255, onTrackbarChange);

    cv::Mat frame, hsv, mask, morphed, vis;
    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        cv::resize(frame, frame, cv::Size(500, 360));

        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(Hmin, Smin, Vmin), cv::Scalar(Hmax, Smax, Vmax), mask);
        cv::Mat kernel = makeKernel(morphK);
        cv::morphologyEx(mask, morphed, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(morphed, morphed, cv::MORPH_CLOSE, kernel);
        cv::cvtColor(morphed, morphed, cv::COLOR_GRAY2BGR);

        vis = frame.clone();
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        int bestIdx = -1;
        double bestScore = 0.0;
        for (int i = 0; i < (int)contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area < minArea) continue;

            double peri = cv::arcLength(contours[i], true);
            if (peri <= 1e-6) continue;

            double circularity = 4.0 * CV_PI * area / (peri * peri);
            if (circularity < 0.35) continue;

            double score = area * circularity;
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }

        if (bestIdx >= 0) {
            cv::Point2f center;
            float radius = 0.f;
            cv::minEnclosingCircle(contours[bestIdx], center, radius);
            cv::circle(vis, center, (int)radius, cv::Scalar(0, 255, 0), 2);
            cv::circle(vis, center, 3, cv::Scalar(0, 255, 0), -1);
            if (drawBoundingBox) {
                cv::Rect rect = cv::boundingRect(contours[bestIdx]);
                cv::rectangle(vis, rect, cv::Scalar(255, 0, 0), 2);
            }
            cv::putText(vis, "Basketball", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(vis, "No ball", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        cv::Mat combined;
        cv::hconcat(frame, morphed, combined);
        cv::hconcat(combined, vis, combined);

        cv::putText(combined, "Basketball Detection | Original + Mask + Result",
                    cv::Point(combined.cols / 2 - 200, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

        cv::imshow("Threshold_debug", combined);

        int key = cv::waitKey(waitTime);
        if (key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
