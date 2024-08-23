#include "grayMatch.h"

#include <opencv2/opencv.hpp>

int main() {
    auto src =
        cv::imread("C:/Users/qiuyong/Desktop/test/template/model3.bmp", cv::IMREAD_GRAYSCALE);
    auto dst =
        cv::imread("C:/Users/qiuyong/Desktop/test/template/model3_src2.bmp", cv::IMREAD_GRAYSCALE);

    auto model = trainModel(src, -1);
    auto poses = matchModel(dst, model, -1, 0, 360, 0, 0.5, 70, 1);

    return 0;
}