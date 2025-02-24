#pragma once

#include <opencv2/core.hpp>

struct HRLE {
    int row         = -1;
    int startColumn = -1;
    int length      = 0;
};

struct VRLE {
    int col      = -1;
    int startRow = -1;
    int length   = 0;
};

using HRegion = std::vector<HRLE>;
using VRegion = std::vector<VRLE>;

struct Template {
    cv::Mat         img;
    HRegion         hRegion;
    VRegion         vRegion;
    cv::RotatedRect rect;

    double mean    = 0;
    double normal  = 0;
    double invArea = 0;
};

struct Layer {
    double angleStep = 0;

    std::vector<Template> templates;
};

struct Model {
    double startAngle = 0;
    double stopAngle  = 0;
    double angleStep  = 0;

    cv::Mat            source;
    std::vector<Layer> layers;
};

#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 8
#define simdSize(type) cv::VTraits<type>::nlanes
#else
#define simdSize(type) type::nlanes
#endif