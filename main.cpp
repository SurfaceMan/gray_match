#include "grayMatch.h"

#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    auto src = cv::imread(std::string(IMG_DIR) + "/model3.png", cv::IMREAD_GRAYSCALE);
    auto dst = cv::imread(std::string(IMG_DIR) + "/model3_src2.png", cv::IMREAD_GRAYSCALE);
    if (src.empty() || dst.empty()) {
        return -1;
    }

    auto t0    = cv::getTickCount();
    auto model = trainModel(src, -1, 0, 360, -1);
    auto t1    = cv::getTickCount();
    auto poses = matchModel(dst, model, -1, 0, 360, 0, 0.5, 70, 1);
    auto t2    = cv::getTickCount();

    const auto trainCost = static_cast<double>(t1 - t0) / cv::getTickFrequency();
    const auto matchCost = static_cast<double>(t2 - t1) / cv::getTickFrequency();
    std::cout << "train(s):" << trainCost << " match(s):" << matchCost << std::endl;

    cv::Mat color;
    cv::cvtColor(dst, color, cv::COLOR_GRAY2RGB);
    for (const auto &pose : poses) {
        cv::RotatedRect rect(cv::Point2f(pose.x, pose.y), src.size(), -pose.angle);

        cv::Point2f pts[ 4 ];
        rect.points(pts);

        cv::line(color, pts[ 0 ], pts[ 1 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::line(color, pts[ 1 ], pts[ 2 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::line(color, pts[ 2 ], pts[ 3 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::line(color, pts[ 3 ], pts[ 0 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

        std::cout << pose.x << "," << pose.y << "," << pose.angle << "," << pose.score << //
            std::endl;
    }

    cv::imshow("img", color);
    cv::waitKey();

    return 0;
}
