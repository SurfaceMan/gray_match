#include "grayMatch.h"

#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    auto src =
        cv::imread("C:/Users/qiuyong/Desktop/test/template/model3.bmp", cv::IMREAD_GRAYSCALE);
    auto dst =
        cv::imread("C:/Users/qiuyong/Desktop/test/template/model3_src1.bmp", cv::IMREAD_GRAYSCALE);

    auto t0    = cv::getTickCount();
    auto model = trainModel(src.data, src.cols, src.rows, src.channels(), int(src.step), -1);
    auto t1    = cv::getTickCount();

    int size;
    serialize(model, nullptr, &size);
    std::vector<uchar> buffer(size);
    serialize(model, buffer.data(), &size);

    freeModel(&model);

    model = deserialize(buffer.data(), int(buffer.size()));

    int               num = 70;
    std::vector<Pose> poses(num);

    auto t2 = cv::getTickCount();
    matchModel(dst.data, dst.cols, dst.rows, dst.channels(), int(dst.step), model, &num,
               poses.data(), -1, 0, 360, 0, 0.5, 1);
    auto t3 = cv::getTickCount();

    auto trainCost = double(t1 - t0) / cv::getTickFrequency();
    auto matchCost = double(t3 - t2) / cv::getTickFrequency();
    std::cout << "train(s):" << trainCost << " match(s):" << matchCost << std::endl;

    cv::Mat color;
    cv::cvtColor(dst, color, cv::COLOR_GRAY2RGB);
    for (int i = 0; i < num; i++) {
        auto           &pose = poses[ i ];
        cv::RotatedRect rect(cv::Point2f(pose.x, pose.y), src.size(), -pose.angle);

        std::vector<cv::Point2f> pts;
        rect.points(pts);

        cv::line(color, pts[ 0 ], pts[ 1 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::line(color, pts[ 1 ], pts[ 2 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::line(color, pts[ 2 ], pts[ 3 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::line(color, pts[ 3 ], pts[ 0 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

        std::cout << pose.x << "," << pose.y << "," << pose.angle << "," << pose.score << std::endl;
    }

    cv::imshow("img", color);
    cv::waitKey();

    return 0;
}
