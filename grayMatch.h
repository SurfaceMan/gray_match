#ifndef GRAY_MATCH_H
#define GRAY_MATCH_H

#include <opencv2/opencv.hpp>

struct Model;

struct Pose {
    float x;
    float y;
    float angle;
    float score;
};

Model *trainModel(const cv::Mat &src, int level, double startAngle, double spanAngle,
                  double anglgStep);

std::vector<Pose> matchModel(const cv::Mat &dst, const Model *model, int level, double startAngle,
                             double spanAngle, double maxOverlap, double minScore, int maxCount,
                             int subpixel);

void serialize(Model *model, int &size, uint8_t *buffer);

Model *deserialize(int size, uint8_t *buffer);

#endif // GRAY_MATCH_H