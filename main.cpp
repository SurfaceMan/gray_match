#include "grayMatch.h"

#include <fstream>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>

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

    auto srcFile = std::string(IMG_DIR) + "/3.bmp";
    auto dstFile = std::string(IMG_DIR) + "/h.bmp";
    if (cmd.has("model"))
        srcFile = cmd.get<std::string>("model");
    if (cmd.has("scene"))
        dstFile = cmd.get<std::string>("scene");

    auto src = cv::imread(srcFile, cv::IMREAD_GRAYSCALE);
    auto dst = cv::imread(dstFile, cv::IMREAD_GRAYSCALE);
    if (src.empty() || dst.empty()) {
        return -1;
    }

    const std::string modelName("model.bin");
    {
        auto t0    = cv::getTickCount();
        auto model = trainModel(src.data, src.cols, src.rows, src.channels(), int(src.step), 0, 0,
                                src.cols, src.rows, -1);
        auto t1    = cv::getTickCount();

        // get size
        int size;
        serialize(model, nullptr, &size);

        // serialize to buffer
        std::vector<uchar> buffer(size);
        serialize(model, buffer.data(), &size);

        // write to file
        std::ofstream ofs(modelName, std::ios::binary | std::ios::out);
        if (!ofs.is_open()) {
            return -1;
        }
        ofs.write((const char *)buffer.data(), size);

        freeModel(&model);

        auto trainCost = double(t1 - t0) / cv::getTickFrequency();
        std::cout << "train(s):" << trainCost << std::endl;
    }

    int               count = 70;
    std::vector<Pose> poses(count);
    Model_t           model;
    auto              score = cmd.get<float>("threshold");
    {
        // open file
        std::ifstream ifs(modelName, std::ios::binary | std::ios::in);
        if (!ifs.is_open()) {
            return -2;
        }

        // get size
        ifs.seekg(0, std::ios::end);
        auto size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        // read to buffer
        std::vector<uchar> buffer(size);
        ifs.read((char *)buffer.data(), size);

        // deserialize from buffer
        model = deserialize(buffer.data(), int(buffer.size()));

        auto t2 = cv::getTickCount();
        matchModel(dst.data, dst.cols, dst.rows, dst.channels(), int(dst.step), 0, 0, dst.cols,
                   dst.rows, model, &count, poses.data(), -1, 0, 360, 0, score, 1);
        auto t3 = cv::getTickCount();

        auto matchCost = double(t3 - t2) / cv::getTickFrequency();
        std::cout << "match(s):" << matchCost << std::endl;
    }

    for (int i = 0; i < count; i++) {
        const auto &pose = poses[ i ];
        std::cout << pose.x << "," << pose.y << "," << pose.angle << "," << pose.score << std::endl;
    }

    if (cmd.has("bench")) {
        const int times = 100;

        auto start = cv::getTickCount();
        for (int i = 0; i < times; i++) {
            matchModel(dst.data, dst.cols, dst.rows, dst.channels(), int(dst.step), 0, 0, dst.cols,
                       dst.rows, model, &count, poses.data(), -1, 0, 360, 0, score, 1);
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
