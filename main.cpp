#include "grayMatch.h"

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    auto src =
        cv::imread("/home/abc/project/ShapeMatch/ShapeMatchTest/TestImage/3.bmp", cv::IMREAD_GRAYSCALE);
    auto dst =
        cv::imread("/home/abc/project/ShapeMatch/ShapeMatchTest/TestImage/l.bmp", cv::IMREAD_GRAYSCALE);

    auto model = trainModel(src, -1);
    auto poses = matchModel(dst, model, 3, 0, 360, 0, 0.5, 70, 1);

    cv::Mat color;
    cv::cvtColor(dst, color, cv::COLOR_GRAY2RGB);
    for(auto &pose : poses){
        cv::RotatedRect rect(cv::Point2f(pose.x, pose.y), src.size(), -pose.angle);

        std::vector<cv::Point2f> pts;
        rect.points(pts);

        cv::line(color, pts[0], pts[1], cv::Scalar(255, 0 , 0), 1, cv::LINE_AA);
        cv::line(color, pts[1], pts[2], cv::Scalar(255, 0 , 0), 1, cv::LINE_AA);
        cv::line(color, pts[2], pts[3], cv::Scalar(255, 0 , 0), 1, cv::LINE_AA);
        cv::line(color, pts[3], pts[0], cv::Scalar(255, 0 , 0), 1, cv::LINE_AA);

        std::cout << pose.x << "," << pose.y << "," << pose.angle << "," <<pose.score <<std::endl;
    }

    cv::imshow("img", color);
    cv::waitKey();

    return 0;
}
