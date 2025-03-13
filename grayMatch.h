#ifndef GRAY_MATCH_H
#define GRAY_MATCH_H

#include <opencv2/opencv.hpp>

#include "apiExport.h"

struct Model;

struct Pose {
    float x;
    float y;
    float angle;
    float score;
};

API_PUBLIC Model *trainModel(const cv::Mat &src, int level, double startAngle, double spanAngle,
                             double angleStep);

API_PUBLIC void matchModel(const cv::Mat &dst, const Model *model, int *count, Pose *poses,
                           int level, double startAngle, double spanAngle, double maxOverlap,
                           double minScore, int subpixel);

#endif // GRAY_MATCH_H