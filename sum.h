#pragma once

#include "privateType.h"

#include <opencv2/opencv.hpp>

void integralSum(const cv::Mat &src, cv::Mat &sum, cv::Mat &sqSum, const cv::Size &templateSize,
                 const HRegion &hRegion, const VRegion &vRegion);
