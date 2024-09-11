#include "grayMatch.h"

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // auto src =
    //     cv::imread("C:/Users/qiuyong/Desktop/test/template/model3.bmp", cv::IMREAD_GRAYSCALE);
    auto dst =
        cv::imread("C:/Users/qiuyong/Desktop/test/template/model3_src2.bmp", cv::IMREAD_GRAYSCALE);

    const std::string modelName("model.bin");
    /*{
        auto t0    = cv::getTickCount();
        auto model = trainModel(src.data, src.cols, src.rows, src.channels(), int(src.step), 0, 0,
                                src.cols, src.rows, -1);
        auto t1    = cv::getTickCount();

        int size;
        serialize(model, nullptr, &size);
        std::vector<uchar> buffer(size);
        serialize(model, buffer.data(), &size);

        std::ofstream ofs(modelName, std::ios::binary | std::ios::out);
        if (!ofs.is_open()) {
            return -1;
        }
        ofs.write((const char *)buffer.data(), size);

        freeModel(&model);

        auto trainCost = double(t1 - t0) / cv::getTickFrequency();
        std::cout << "train(s):" << trainCost << std::endl;
    }*/

    int               num = 70;
    std::vector<Pose> poses(num);
    {
        std::ifstream ifs(modelName, std::ios::binary | std::ios::in);
        if (!ifs.is_open()) {
            return -2;
        }
        ifs.seekg(0, std::ios::end);
        auto size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::vector<uchar> buffer(size);
        ifs.read((char *)buffer.data(), size);

        auto model = deserialize(buffer.data(), int(buffer.size()));

        auto t2 = cv::getTickCount();
        matchModel(dst.data, dst.cols, dst.rows, dst.channels(), int(dst.step), 0, 0, dst.cols,
                   dst.rows, model, &num, poses.data(), 3, 0, 360, 0, 0.5, 1);
        auto t3 = cv::getTickCount();

        auto matchCost = double(t3 - t2) / cv::getTickFrequency();
        std::cout << "match(s):" << matchCost << std::endl;
    }

    // cv::Mat color;
    // cv::cvtColor(dst, color, cv::COLOR_GRAY2RGB);
    for (int i = 0; i < num; i++) {
        auto &pose = poses[ i ];
        // cv::RotatedRect rect(cv::Point2f(pose.x, pose.y), src.size(), -pose.angle);
        //
        // cv::Point2f pts[ 4 ];
        // rect.points(pts);
        //
        // cv::line(color, pts[ 0 ], pts[ 1 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        // cv::line(color, pts[ 1 ], pts[ 2 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        // cv::line(color, pts[ 2 ], pts[ 3 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        // cv::line(color, pts[ 3 ], pts[ 0 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

        std::cout << pose.x << "," << pose.y << "," << pose.angle << "," << pose.score << std::endl;
    }

    // cv::imshow("img", color);
    // cv::waitKey();

    return 0;
}
