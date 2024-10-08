#pragma once

#include <opencv2/core.hpp>

struct Model {
    std::vector<cv::Mat>    pyramids;
    std::vector<cv::Scalar> mean;
    std::vector<double>     normal;
    std::vector<double>     invArea;
    std::vector<uchar>      equal1;
    uchar                   borderColor = 0;

    void clear() {
        pyramids.clear();
        normal.clear();
        invArea.clear();
        mean.clear();
        equal1.clear();
    }

    void resize(const std::size_t size) {
        normal.resize(size);
        invArea.resize(size);
        mean.resize(size);
        equal1.resize(size);
    }

    void reserve(const std::size_t size) {
        pyramids.reserve(size);
        normal.reserve(size);
        invArea.reserve(size);
        mean.reserve(size);
        equal1.reserve(size);
    }
};

#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 8
#define simdSize(type) cv::VTraits<type>::nlanes
#else
#define simdSize(type) type::nlanes
#endif
