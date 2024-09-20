#ifndef INTEGRAL_H
#define INTEGRAL_H

#include <opencv2/opencv.hpp>

void integralSimd(const cv::Mat &src, cv::Mat &sum, cv::Mat &sqSum);

#endif // INTEGRAL_H
