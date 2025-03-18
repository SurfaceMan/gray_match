#include "grayMatch.h"

#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <string>

int main(int argc, const char *argv[]) {
    const std::string keys = "{model m || model image}"
                             "{scene s || scene image}"
                             "{view v || view result}"
                             "{threshold t | 0.7 | match minium score}"
                             "{bench b || match bechmark}"
                             "{help h || print this help}";

    cv::CommandLineParser cmd(argc, argv, keys);
    if (!cmd.check()) {
        cmd.printErrors();
        return -1;
    }

    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    auto srcFile = std::string(IMG_DIR) + "/model3.png";
    auto dstFile = std::string(IMG_DIR) + "/model3_src2.png";
    if (cmd.has("model"))
        srcFile = cmd.get<std::string>("model");
    if (cmd.has("scene"))
        dstFile = cmd.get<std::string>("scene");

    auto src = cv::imread(srcFile, cv::IMREAD_GRAYSCALE);
    auto dst = cv::imread(dstFile, cv::IMREAD_GRAYSCALE);
    if (src.empty() || dst.empty()) {
        return -1;
    }

    int               count = 70;
    std::vector<Pose> poses(count);
    auto              score = cmd.get<float>("threshold");

    auto t0    = cv::getTickCount();
    auto model = trainModel(src, -1, 0, 360, -1);
    auto t1    = cv::getTickCount();
    matchModel(dst, model, &count, poses.data(), -1, 0, 360, 0, score, 1);
    auto t2 = cv::getTickCount();

    const auto trainCost = static_cast<double>(t1 - t0) / cv::getTickFrequency();
    const auto matchCost = static_cast<double>(t2 - t1) / cv::getTickFrequency();
    std::cout << "train(s):" << trainCost << " match(s):" << matchCost << std::endl;
    for (int i = 0; i < count; i++) {
        const auto &pose = poses[ i ];
        std::cout << pose.x << "," << pose.y << "," << pose.angle << "," << pose.score << std::endl;
    }

    if (cmd.has("bench")) {
        const int times = 100;

        auto start = cv::getTickCount();
        for (int i = 0; i < times; i++) {
            matchModel(dst, model, &count, poses.data(), -1, 0, 360, 0, score, 1);
            count = 70;
        }
        auto end = cv::getTickCount();

        const auto cost = static_cast<double>(end - start) / cv::getTickFrequency() / times;
        std::cout << "match bench avg(s):" << cost << std::endl;
    }

    if (cmd.has("view")) {
        cv::Mat color;
        cv::cvtColor(dst, color, cv::COLOR_GRAY2RGB);
        for (int i = 0; i < count; i++) {
            const auto     &pose = poses[ i ];
            cv::RotatedRect rect(cv::Point2f(pose.x, pose.y), src.size(), -pose.angle);

            cv::Point2f pts[ 4 ];
            rect.points(pts);

            cv::line(color, pts[ 0 ], pts[ 1 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            cv::line(color, pts[ 1 ], pts[ 2 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            cv::line(color, pts[ 2 ], pts[ 3 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            cv::line(color, pts[ 3 ], pts[ 0 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        }

        cv::imshow("img", color);
        cv::waitKey();
    }

    return 0;
}
