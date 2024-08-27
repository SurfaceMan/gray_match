#include "grayMatch.h"

#include <complex.h>
#include <opencv2/core/hal/intrin.hpp>
#include <utility>

constexpr int    MIN_AREA  = 256;
constexpr double TOLERANCE = 0.0000001;
constexpr int    CANDIDATE = 5;
constexpr double INVALID   = -1.;

struct Model {
    std::vector<cv::Mat>    pyramids;
    std::vector<cv::Scalar> mean;
    std::vector<double>     normal;
    std::vector<double>     invArea;
    std::vector<bool>       equal1;
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
        normal.reserve(size);
        invArea.reserve(size);
        mean.reserve(size);
        equal1.reserve(size);
    }
};

struct BlockMax {
    struct Block {
        cv::Rect  rect;
        cv::Point maxPos;
        double    maxScore;

        Block()
            : maxScore(0) {}

        Block(const cv::Rect &rect_, const cv::Point &maxPos_, const double maxScore_)
            : rect(rect_)
            , maxPos(maxPos_)
            , maxScore(maxScore_) {}

        Block(Block &&rhs) noexcept {
            maxScore = rhs.maxScore;
            maxPos   = rhs.maxPos;
            maxScore = rhs.maxScore;
        }

        bool operator<(const Block &rhs) const {
            return this->maxScore > rhs.maxScore;
        }
    };

    std::vector<Block> blocks;
    cv::Mat            score;

    BlockMax(cv::Mat score_, cv::Size templateSize) {
        score = std::move(score_);

        // divide source image to blocks then compute max
        auto blockWidth  = templateSize.width * 2;
        auto blockHeight = templateSize.height * 2;

        auto nWidth    = score.size().width / blockWidth;
        auto nHeight   = score.size().height / blockHeight;
        auto hRemained = score.size().width % blockWidth;
        auto vRemained = score.size().height % blockHeight;

        blocks.reserve(nWidth * nHeight);
        for (int y = 0; y < nHeight; y++) {
            for (int x = 0; x < nWidth; x++) {
                cv::Rect rect(x * blockWidth, y * blockHeight, blockWidth, blockHeight);

                Block block;
                block.rect = rect;
                cv::minMaxLoc(score(rect), nullptr, &block.maxScore, nullptr, &block.maxPos);
                block.maxPos += rect.tl();
                blocks.push_back(std::move(block));
            }
        }

        if (hRemained) {
            cv::Rect rightRect(nWidth * blockWidth, 0, hRemained, score.size().height);
            Block    rightBlock;
            rightBlock.rect = rightRect;
            cv::minMaxLoc(score(rightRect), nullptr, &rightBlock.maxScore, nullptr,
                          &rightBlock.maxPos);
            rightBlock.maxPos += rightRect.tl();
            blocks.push_back(std::move(rightBlock));
        }

        if (vRemained) {
            auto width = hRemained ? nWidth * blockWidth : score.size().width;
            if (width < 1) {
                return;
            }

            cv::Rect bottomRect(0, nHeight * blockHeight, width, vRemained);
            Block    bottomBlock;
            bottomBlock.rect = bottomRect;
            cv::minMaxLoc(score(bottomRect), nullptr, &bottomBlock.maxScore, nullptr,
                          &bottomBlock.maxPos);
            bottomBlock.maxPos += bottomRect.tl();
            blocks.push_back(std::move(bottomBlock));
        }
    }

    void update(const cv::Rect &rect) {
        for (auto &block : blocks) {
            auto intersection = block.rect & rect;
            if (intersection.empty()) {
                continue;
            }

            // update
            cv::minMaxLoc(score(block.rect), nullptr, &block.maxScore, nullptr, &block.maxPos);
            block.maxPos += block.rect.tl();
        }
    }

    void maxValueLoc(double &maxScore, cv::Point &maxPos) {
        const auto max = std::max_element(blocks.begin(), blocks.end());
        maxScore       = max->maxScore;
        maxPos         = max->maxPos;
    }
};

struct Candidate {
    cv::Point2d pos;
    double      angle;
    double      score;

    Candidate()
        : angle(0)
        , score(0) {}

    Candidate(const cv::Point2d &pos_, const double angle_, const double score_)
        : pos(pos_)
        , angle(angle_)
        , score(score_) {}

    bool operator<(const Candidate &rhs) const {
        return this->score > rhs.score;
    }
};

struct RLE {
    int row         = -1;
    int startColumn = -1;
    int length      = 0;
};

using Region = std::vector<RLE>;

struct Template {
    cv::Mat     img;
    Region      region;
    cv::Point2d center;
    double      angle = 0;
};

struct Layer {
    double angleStep = 0;
    double mean      = 0;
    double normal    = 0;
    double invArea   = 0;

    std::vector<Template> templates;
};

struct Model2 {
    double startAngle  = 0;
    double stopAngle   = 0;
    double angleStep   = 0;
    uchar  borderColor = 0;

    cv::Mat            source;
    std::vector<Layer> layers;
};

inline cv::Point2d transform(const cv::Point2d &point, const cv::Mat &rotate) {
    const auto ptr = rotate.ptr<double>();

    auto x = point.x * ptr[ 0 ] + point.y * ptr[ 1 ] + ptr[ 2 ];
    auto y = point.x * ptr[ 3 ] + point.y * ptr[ 4 ] + ptr[ 5 ];

    return {x, y};
}

inline cv::Point2d transform(const cv::Point2d &point, const cv::Point &center, double angle) {
    const auto rotate = cv::getRotationMatrix2D(center, angle, 1.);

    return transform(point, rotate);
}

inline cv::Point2d sizeCenter(const cv::Size &size) {
    return {(size.width - 1.) / 2., (size.height - 1.) / 2.};
}

inline cv::Point2d sizeCenter(const cv::Size2d &size) {
    return {(size.width - 1.) / 2., (size.height - 1.) / 2.};
}

inline double sizeAngleStep(const cv::Size &size) {
    const auto diameter = sqrt(size.width * size.width + size.height * size.height);
    return atan(2. / diameter) * 180. / CV_PI;
}

int computeLayers(const int width, const int height, const int minArea) {
    assert(width > 0 && height > 0 && minArea > 0);

    auto area  = width * height;
    int  layer = 0;
    while (area > minArea) {
        area /= 4;
        layer++;
    }

    return layer;
}

cv::Rect2d boundingRect(const std::vector<cv::Point2d> &points) {
    if (points.empty()) {
        return {};
    }

    cv::Point2d min = points[ 0 ];
    cv::Point2d max = points[ 0 ];
    for (const auto &point : points) {
        if (point.x < min.x) {
            min.x = point.x;
        }
        if (point.y < min.y) {
            min.y = point.y;
        }

        if (point.x > max.x) {
            max.x = point.x;
        }
        if (point.y > max.y) {
            max.y = point.y;
        }
    }

    return {min, max};
}

cv::Size computeRotationSize(const cv::Size &dstSize, const cv::Size &templateSize, double angle,
                             const cv::Mat &rotate) {
    if (angle > 360) {
        angle -= 360;
    } else if (angle < 0) {
        angle += 360;
    }

    if (fabs(fabs(angle) - 90) < TOLERANCE || fabs(fabs(angle) - 270) < TOLERANCE) {
        return {dstSize.height, dstSize.width};
    }

    if (fabs(angle) < TOLERANCE || fabs(fabs(angle) - 180) < TOLERANCE) {
        return dstSize;
    }

    const std::vector<cv::Point2d> points{
        {0, 0},
        {static_cast<double>(dstSize.width) - 1, 0},
        {0, static_cast<double>(dstSize.height) - 1},
        {static_cast<double>(dstSize.width) - 1, static_cast<double>(dstSize.height) - 1}};
    std::vector<cv::Point2d> trans;
    cv::transform(points, trans, rotate);
    const auto rect = boundingRect(trans);
    const auto max  = rect.br();
    const auto min  = rect.tl();

    if (angle > 0 && angle < 90) {
        ;
    } else if (angle > 90 && angle < 180) {
        angle -= 90;
    } else if (angle > 180 && angle < 270) {
        angle -= 180;
    } else if (angle > 270 && angle < 360) {
        angle -= 270;
    }

    const auto radius = angle / 180. * CV_PI;
    const auto dy     = sin(radius);
    const auto dx     = cos(radius);
    const auto width  = templateSize.width * dx * dy;
    const auto height = templateSize.height * dx * dy;

    const auto center     = sizeCenter(dstSize);
    const auto halfHeight = static_cast<int>(ceil(max.y - center.y - width));
    const auto halfWidth  = static_cast<int>(ceil(max.x - center.x - height));

    cv::Size   size(halfWidth * 2, halfHeight * 2);
    const auto wrongSize = (templateSize.width < size.width && templateSize.height > size.height) ||
                           (templateSize.width > size.width && templateSize.height < size.height) ||
                           templateSize.area() > size.area();
    if (wrongSize) {
        size = {static_cast<int>(max.x - min.x + 0.5), static_cast<int>(max.y - min.y + 0.5)};
    }

    return size;
}

void coeffDenominator(const cv::Mat &src, const cv::Size &templateSize, cv::Mat &result,
                      const double mean, const double normal, const double invArea,
                      const bool equal1) {
    if (equal1) {
        result = cv::Scalar::all(1);
        return;
    }

    cv::Mat sum;
    cv::Mat sqSum;
    cv::integral(src, sum, sqSum, CV_64F);

    const auto *q0 = sqSum.ptr<double>(0);
    const auto *q1 = q0 + templateSize.width;
    const auto *q2 = sqSum.ptr<double>(templateSize.height);
    const auto *q3 = q2 + templateSize.width;

    const auto *p0 = sum.ptr<double>(0);
    const auto *p1 = p0 + templateSize.width;
    const auto *p2 = sum.ptr<double>(templateSize.height);
    const auto *p3 = p2 + templateSize.width;

    const auto step = sum.step / sizeof(double);

    for (int y = 0; y < result.rows; y++) {
        auto *scorePtr = result.ptr<float>(y);
        auto  idx      = y * step;
        for (int x = 0; x < result.cols; x++, idx++) {
            auto      &score    = scorePtr[ x ];
            const auto partSum  = p0[ idx ] - p1[ idx ] - p2[ idx ] + p3[ idx ];
            auto       partMean = partSum * partSum * invArea;
            const auto num      = score - partSum * mean;

            auto partSum2 = q0[ idx ] - q1[ idx ] - q2[ idx ] + q3[ idx ];

            const auto diff = std::max(partSum2 - partMean, 0.);
            if (diff <= std::min(0.5, 10 * FLT_EPSILON * partSum2)) {
                partSum2 = 0;
            } else {
                partSum2 = sqrt(diff) * normal;
            }

            if (abs(num) < partSum2) {
                score = static_cast<float>(num / partSum2);
            } else if (abs(num) < partSum2 * 1.125) {
                score = num > 0.f ? 1.f : -1.f;
            } else {
                score = 0;
            }
        }
    }
}

#ifdef CV_SIMD
float convSimd(const uchar *kernel, const uchar *src, const int kernelWidth) {
    const auto blockSize = cv::VTraits<cv::v_uint8>::vlanes();
    auto       vSum      = cv::vx_setall_u32(0);
    int        i         = 0;
    for (; i < kernelWidth - blockSize; i += blockSize) {
        vSum += cv::v_dotprod_expand(cv::v_load(kernel + i), cv::v_load(src + i));
    }
    auto sum = cv::v_reduce_sum(vSum);

    for (; i < kernelWidth; i++) {
        sum += kernel[ i ] * src[ i ];
    }

    return static_cast<float>(sum);
}

void computeLine(const uchar *kernel, const uchar *src, const int kernelWidth, uint64_t &dot,
                 uint64_t &sum, uint64_t &sqSum) {
    const auto blockSize = cv::VTraits<cv::v_uint8>::vlanes();
    auto       vDot      = cv::vx_setall_u32(0);
    auto       vsqSum    = cv::vx_setall_u32(0);
    int        i         = 0;
    for (; i < kernelWidth - blockSize; i += blockSize) {
        auto vSrc  = cv::v_load(src + i);
        sum       += cv::v_reduce_sum(vSrc);
        vsqSum    += cv::v_dotprod_expand(vSrc, vSrc);
        vDot      += cv::v_dotprod_expand(cv::v_load(kernel + i), vSrc);
    }
    dot   += cv::v_reduce_sum(vDot);
    sqSum += cv::v_reduce_sum(vsqSum);

    for (; i < kernelWidth; i++) {
        dot   += kernel[ i ] * src[ i ];
        sum   += src[ i ];
        sqSum += src[ i ] * src[ i ];
    }
}

void matchTemplateSimd(const cv::Mat &src, const cv::Mat &templateImg, cv::Mat &result) {
    result = cv::Mat::zeros(src.size() - templateImg.size() + cv::Size(1, 1), CV_32FC1);

    for (int y = 0; y < result.rows; y++) {
        auto *resultPtr = result.ptr<float>(y);
        for (int x = 0; x < result.cols; x++) {
            auto &score = resultPtr[ x ];
            for (int templateRow = 0; templateRow < templateImg.rows; templateRow++) {
                auto *srcPtr  = src.ptr<uchar>(y + templateRow) + x;
                auto *temPtr  = templateImg.ptr<uchar>(templateRow);
                score        += convSimd(temPtr, srcPtr, templateImg.cols);
            }
        }
    }
}

void matchTemplateSimd(const cv::Mat &src, const cv::Mat &templateImg, const Region &region,
                       cv::Mat &result, const double mean, const double normal,
                       const double invArea) {
    result = cv::Mat::zeros(src.size() - templateImg.size() + cv::Size(1, 1), CV_32FC1);

    for (int y = 0; y < result.rows; y++) {
        auto *resultPtr = result.ptr<float>(y);
        for (int x = 0; x < result.cols; x++) {
            auto    &score = resultPtr[ x ];
            uint64_t dot   = 0;
            uint64_t sum   = 0;
            uint64_t sqSum = 0;
            for (const auto &rle : region) {
                auto *srcPtr = src.ptr<uchar>(y + rle.row) + x;
                auto *temPtr = templateImg.ptr<uchar>(rle.row) + rle.startColumn;
                computeLine(temPtr, srcPtr, rle.length, dot, sum, sqSum);
            }

            const auto partMean = sum * sum * invArea;
            const auto num      = dot - sum * mean;

            const auto diff   = std::max(sqSum - partMean, 0.);
            auto       fsqSum = static_cast<double>(sqSum);
            if (diff <= std::min(0.5, 10 * FLT_EPSILON * fsqSum)) {
                fsqSum = 0;
            } else {
                fsqSum = sqrt(diff) * normal;
            }

            if (abs(num) < fsqSum) {
                score = static_cast<float>(num / fsqSum);
            } else if (abs(num) < fsqSum * 1.125) {
                score = num > 0.f ? 1.f : -1.f;
            } else {
                score = 0;
            }
        }
    }
}

void matchTemplate(const cv::Mat &src, cv::Mat &result, const Template &layerTemplate,
                   const double mean, const double normal, const double invArea) {
    if (layerTemplate.region.empty()) {
        matchTemplateSimd(src, layerTemplate.img, result);
        coeffDenominator(src, layerTemplate.img.size(), result, mean, normal, invArea, false);
    } else {
        matchTemplateSimd(src, layerTemplate.img, layerTemplate.region, result, mean, normal,
                          invArea);
    }
}
#endif

void matchTemplate(cv::Mat &src, cv::Mat &result, const Model *model, int level) {
#ifdef CV_SIMD
    matchTemplateSimd(src, model->pyramids[ level ], result);
#else
    cv::matchTemplate(src, model->pyramids[ level ], result, cv::TM_CCORR);
#endif
    coeffDenominator(src, model->pyramids[ level ].size(), result, model->mean[ level ][ 0 ],
                     model->normal[ level ], model->invArea[ level ], model->equal1[ level ]);
}

void nextMaxLoc(const cv::Point &pos, const cv::Size templateSize, const double maxOverlap,
                BlockMax &block, double &maxScore, cv::Point &maxPos) {
    const auto      alone = 1. - maxOverlap;
    const cv::Point offset(static_cast<int>(templateSize.width * alone),
                           static_cast<int>(templateSize.height * alone));
    const cv::Size  size(static_cast<int>(2 * templateSize.width * alone),
                         static_cast<int>(2 * templateSize.height * alone));
    const cv::Rect  rectIgnore(pos - offset, size);

    // clear neighbor
    cv::rectangle(block.score, rectIgnore, cv::Scalar(-1), cv::FILLED);

    block.update(rectIgnore);
    block.maxValueLoc(maxScore, maxPos);
}

void nextMaxLoc(cv::Mat &score, const cv::Point &pos, const cv::Size templateSize,
                const double maxOverlap, double &maxScore, cv::Point &maxPos) {
    const auto      alone = 1. - maxOverlap;
    const cv::Point offset(static_cast<int>(templateSize.width * alone),
                           static_cast<int>(templateSize.height * alone));
    const cv::Size  size(static_cast<int>(2 * templateSize.width * alone),
                         static_cast<int>(2 * templateSize.height * alone));
    const cv::Rect  rectIgnore(pos - offset, size);

    // clear neighbor
    cv::rectangle(score, rectIgnore, cv::Scalar(-1), cv::FILLED);

    cv::minMaxLoc(score, nullptr, &maxScore, nullptr, &maxPos);
}

void cropRotatedRoi(const cv::Mat &src, const cv::Size &templateSize, const cv::Point2d topLeft,
                    const cv::Mat &rotate, cv::Mat &roi) {
    const auto     point = transform(topLeft, rotate);
    const cv::Size paddingSize(templateSize.width + 6, templateSize.height + 6);
    auto           rt    = rotate;
    rt.at<double>(0, 2) -= point.x - 3;
    rt.at<double>(1, 2) -= point.y - 3;

    cv::warpAffine(src, roi, rt, paddingSize);
}

void filterOverlap(std::vector<Candidate> &candidates, const std::vector<cv::RotatedRect> &rects,
                   const double maxOverlap) {
    const auto size = candidates.size();
    for (std::size_t i = 0; i < size; i++) {
        auto &candidate = candidates[ i ];
        auto &rect      = rects[ i ];
        if (candidate.score < 0) {
            continue;
        }

        for (std::size_t j = i + 1; j < size; j++) {
            auto &refCandidate = candidates[ j ];
            auto &refRect      = rects[ j ];
            if (refCandidate.score < 0) {
                continue;
            }

            std::vector<cv::Point2f> points;
            const auto               type = cv::rotatedRectangleIntersection(rect, refRect, points);

            switch (type) {
                case cv::INTERSECT_NONE: {
                    continue;
                }
                case cv::INTERSECT_FULL: {
                    (candidate.score > refCandidate.score ? refCandidate.score : candidate.score) =
                        INVALID;
                    break;
                }
                case cv::INTERSECT_PARTIAL: {
                    if (points.size() < 2) {
                        continue;
                    }

                    const auto area    = cv::contourArea(points);
                    const auto overlap = area / rect.size.area();
                    if (overlap > maxOverlap) {
                        (candidate.score > refCandidate.score ? refCandidate.score
                                                              : candidate.score) = INVALID;
                    }
                }
                default:;
            }
        }
    }
}

Region borderRegion(const cv::Mat &src) {
    Region result;

    for (int y = 0; y < src.rows; y++) {
        auto *ptr    = src.ptr<uchar>(y);
        bool  inside = false;
        RLE   rle;
        for (int x = 0; x < src.cols; x++) {
            if (0 == ptr[ x ]) {
                continue;
            }

            if (!inside) {
                inside          = true;
                rle.row         = y;
                rle.startColumn = x;
                rle.length      = 1;
                continue;
            }

            rle.length++;
            if (0 != ptr[ x - 1 ]) {
                continue;
            }

            rle.length = x - rle.startColumn + 1;
            break;
        }

        if (inside) {
            result.push_back(rle);
        }
    }

    return result;
}

void drawRegion(cv::Mat &src, const Region &region) {
    for (const auto &rle : region) {
        auto *ptr = src.ptr<uchar>(rle.row) + rle.startColumn;
        memset(ptr, 255, rle.length);
    }
}

double normalizeAngle(const double angle) {
    if (angle < 0.0) {
        const int k = static_cast<int>(std::ceil(-angle / 360.0));
        return angle + k * 360.0;
    }

    const int k = static_cast<int>(std::floor(angle / 360.0));
    return angle - k * 360.0;
}

Model2 *trainModel(const cv::Mat &src, int level, double startAngle, double spanAngle,
                   double angleStep) {
    if (src.empty() || src.channels() != 1) {
        return nullptr;
    }

    if (level < 0) {
        // level must greater than 1
        level = computeLayers(src.size().width, src.size().height, MIN_AREA);
    }

    const auto scale   = 1 << (level - 1);
    const auto topArea = src.size().area() / (scale * scale);
    if (MIN_AREA > topArea) {
        // top area must greater than MIN_AREA
        return nullptr;
    }

    if (angleStep < 0) {
        auto size     = src.size();
        auto diameter = sqrt(size.width * size.width + size.height * size.height);
        angleStep     = diameter < 200 ? 1 : atan(2. / diameter) * 180. / CV_PI;
    }

    if (spanAngle <= 0) {
        return nullptr;
    }
    if (spanAngle > 360) {
        spanAngle = 360;
    }
    startAngle  = normalizeAngle(startAngle);
    auto nAngle = static_cast<int>(ceil(spanAngle / angleStep));

    auto   *result   = new Model2;
    Model2 &model    = *result;
    model.startAngle = startAngle;
    model.angleStep  = angleStep;
    model.stopAngle  = startAngle + angleStep * nAngle;
    model.source     = src;

    std::vector<cv::Mat> pyramids;
    cv::buildPyramid(src, pyramids, level - 1);
    for (int i = 0; i < level; i++) {
        const auto &pyramid = pyramids[ i ];
        cv::Scalar  mean;
        cv::Scalar  stdDev;
        cv::meanStdDev(pyramid, mean, stdDev);
        const auto &stdDevVal    = stdDev[ 0 ];
        const auto  stdNormal    = stdDevVal * stdDevVal;
        auto        noDifference = stdNormal < 1.;
        if (noDifference) {
            // all same grey value
            break;
        }

        const auto area = pyramid.size().area();
        Layer      layer;
        layer.invArea   = 1. / area;
        layer.normal    = stdDevVal / sqrt(layer.invArea);
        layer.angleStep = angleStep * (1 << i);
        layer.mean      = mean[ 0 ];

        auto count  = static_cast<int>(ceil(spanAngle / layer.angleStep)) + 1;
        auto center = sizeCenter(pyramid.size());

        std::vector<cv::Point2d> points{{0., 0.},
                                        {pyramid.cols - 1., 0.},
                                        {pyramid.cols - 1., pyramid.rows - 1.},
                                        {0., pyramid.rows - 1.}};
        for (int j = 0; j < count; j++) {
            Template layerTemplate;

            layerTemplate.angle = startAngle + layer.angleStep * j;
            auto rotate         = cv::getRotationMatrix2D(center, layerTemplate.angle, 1.0);

            std::vector<cv::Point2d> trans;
            cv::transform(points, trans, rotate);
            auto rect = boundingRect(trans);
            auto rectSize =
                cv::Size(static_cast<int>(ceil(rect.width)), static_cast<int>(ceil(rect.height)));
            layerTemplate.center = sizeCenter(rectSize);

            auto offset              = layerTemplate.center - center;
            rotate.at<double>(0, 2) += offset.x;
            rotate.at<double>(1, 2) += offset.y;
            cv::warpAffine(pyramid, layerTemplate.img, rotate, rectSize, cv::INTER_LINEAR,
                           cv::BORDER_DEFAULT);

            // if background area less than 1 percent, just match rotated image
            auto rate = static_cast<double>(area) / rectSize.area();
            if (rate > 0.9) {
                layer.templates.emplace_back(std::move(layerTemplate));
                continue;
            }

            cv::Mat roi = cv::Mat::zeros(rectSize, CV_8UC1);
            cv::line(roi, trans[ 0 ] + offset, trans[ 1 ] + offset, 255, 1, cv::LINE_8);
            cv::line(roi, trans[ 1 ] + offset, trans[ 2 ] + offset, 255, 1, cv::LINE_8);
            cv::line(roi, trans[ 2 ] + offset, trans[ 3 ] + offset, 255, 1, cv::LINE_8);
            cv::line(roi, trans[ 3 ] + offset, trans[ 0 ] + offset, 255, 1, cv::LINE_8);

            // draw rotated rect line, then encode with run-length-encode
            layerTemplate.region = borderRegion(roi);

            layer.templates.emplace_back(std::move(layerTemplate));
        }

        model.layers.emplace_back(std::move(layer));
    }

    if (!model.layers.empty()) {
        model.borderColor = model.layers.front().mean > 128 ? 0 : 255;
    }

    return result;
}

std::vector<Candidate> matchTopLevel(const cv::Mat &dstTop, const Layer &layer,
                                     const double modelAngleStart, double startAngle,
                                     double spanAngle, const double maxOverlap,
                                     const double minScore, const int maxCount, const int level) {
    const int startIndex =
        static_cast<int>(floor((startAngle - modelAngleStart) / layer.angleStep));
    const int count = static_cast<int>(ceil(spanAngle / layer.angleStep)) + 1;

    std::vector<Candidate> candidates;

    const auto topScoreThreshold = minScore * pow(0.9, level);
    const bool calMaxByBlock =
        (dstTop.size().area() / layer.templates.front().img.size().area() > 500) && maxCount > 10;
    for (int i = 0; i < count; i++) {
        const auto &layerTemplate = layer.templates[ startIndex + i ];

        cv::Mat result;
        matchTemplate(dstTop, result, layerTemplate, layer.mean, layer.normal, layer.invArea);

        if (calMaxByBlock) {
            BlockMax  block(result, layerTemplate.img.size());
            double    maxScore;
            cv::Point maxPos;
            block.maxValueLoc(maxScore, maxPos);
            if (maxScore < topScoreThreshold) {
                continue;
            }

            candidates.emplace_back(cv::Point2d(maxPos) + layerTemplate.center, layerTemplate.angle,
                                    maxScore);
            for (int j = 0; j < maxCount + CANDIDATE - 1; j++) {
                nextMaxLoc(maxPos, layerTemplate.img.size(), maxOverlap, block, maxScore, maxPos);
                if (maxScore < topScoreThreshold) {
                    break;
                }

                candidates.emplace_back(cv::Point2d(maxPos) + layerTemplate.center,
                                        layerTemplate.angle, maxScore);
            }
        } else {
            double    maxScore;
            cv::Point maxPos;
            cv::minMaxLoc(result, nullptr, &maxScore, nullptr, &maxPos);
            if (maxScore < topScoreThreshold) {
                continue;
            }

            candidates.emplace_back(cv::Point2d(maxPos) + layerTemplate.center, layerTemplate.angle,
                                    maxScore);
            for (int j = 0; j < maxCount + CANDIDATE - 1; j++) {
                nextMaxLoc(result, maxPos, layerTemplate.img.size(), maxOverlap, maxScore, maxPos);
                if (maxScore < topScoreThreshold) {
                    break;
                }

                candidates.emplace_back(cv::Point2d(maxPos) + layerTemplate.center,
                                        layerTemplate.angle, maxScore);
            }
        }
    }
    std::sort(candidates.begin(), candidates.end());

    return candidates;
}

std::vector<Candidate> matchDownLevel(const std::vector<cv::Mat>   &pyramids,
                                      const double                  modelAngleStart,
                                      const std::vector<Candidate> &candidates,
                                      const double minScore, const int subpixel,
                                      const Model2 *model, int level) {
    std::vector<Candidate> levelMatched;

    for (const auto &candidate : candidates) {
        auto pose    = candidate;
        bool matched = true;
        for (int currentLevel = level - 1; currentLevel >= 0; currentLevel--) {
            const auto &layer          = model->layers[ currentLevel ];
            const int   anglendex      = cvRound((pose.angle - modelAngleStart) / layer.angleStep);
            const auto &currentDstImg  = pyramids[ currentLevel ];
            const auto  center         = pose.pos * 2.;
            const auto  scoreThreshold = minScore * pow(0.9, currentLevel);

            Candidate newCandidate;
            cv::Mat   newScoreRect;
            for (int i = -1; i <= 1; i++) {
                auto index = anglendex + i;
                if (index < 0 || index >= layer.templates.size()) {
                    // out of range
                    continue;
                }

                // out of range?
                const auto &layerTemplate = layer.templates[ index ];
                auto offset = cv::Point2d(layerTemplate.img.cols / 2., layerTemplate.img.rows / 2.);
                cv::Rect rect(center - offset - cv::Point2d(3, 3),
                              center + offset + cv::Point2d(3, 3));
                if (rect.x < 0) {
                    rect.x = 0;
                }
                if (rect.y < 0) {
                    rect.y = 0;
                }
                auto outside = rect.x + rect.width - (currentDstImg.cols - 1);
                if (outside > 0) {
                    rect.width -= outside;
                }
                outside = rect.y + rect.height - (currentDstImg.rows - 1);
                if (outside > 0) {
                    rect.height -= outside;
                }

                cv::Mat result;
                matchTemplate(currentDstImg(rect), result, layerTemplate, layer.mean, layer.normal,
                              layer.invArea);

                double    maxScore;
                cv::Point maxPos;
                cv::minMaxLoc(result, nullptr, &maxScore, nullptr, &maxPos);

                if (newCandidate.score >= maxScore || maxScore < scoreThreshold) {
                    continue;
                }

                newCandidate = {cv::Point2d(maxPos) + cv::Point2d(rect.tl()) + layerTemplate.center,
                                layerTemplate.angle, maxScore};
                if (0 == currentLevel && 1 == subpixel) {
                    auto isBorder = 0 == maxPos.x || 0 == maxPos.y || result.cols - 1 == maxPos.x ||
                                    result.rows - 1 == maxPos.y;
                    newScoreRect = isBorder
                                       ? cv::Mat()
                                       : result(cv::Rect(maxPos - cv::Point{1, 1}, cv::Size(3, 3)));
                }
            }

            if (newCandidate.score < scoreThreshold) {
                matched = false;
                break;
            }

            if (!newScoreRect.empty()) {
                // TODO subpixel
            }

            pose = newCandidate;
        }

        if (!matched) {
            continue;
        }

        levelMatched.push_back(pose);
    }
    std::sort(levelMatched.begin(), levelMatched.end());

    return levelMatched;
}

std::vector<Pose> matchModel(const cv::Mat &dst, const Model2 *model, int level, double startAngle,
                             double spanAngle, const double maxOverlap, const double minScore,
                             const int maxCount, const int subpixel) {
    (void)(subpixel);

    // prepare
    {
        if (dst.empty() || nullptr == model || model->layers.empty()) {
            return {};
        }

        auto &templateImg = model->source;
        if (dst.cols < templateImg.cols || dst.rows < templateImg.rows ||
            dst.size().area() < templateImg.size().area()) {
            return {};
        }

        // TODO angle cross 0/360
        if (spanAngle <= 0) {
            return {};
        }
        if (spanAngle > 360) {
            spanAngle = 360;
        }
        startAngle     = normalizeAngle(startAngle);
        auto stopAngle = startAngle + spanAngle;
        if (startAngle < model->startAngle) {
            startAngle = model->startAngle;
        }
        if (stopAngle > model->stopAngle) {
            stopAngle = model->stopAngle;
        }
        spanAngle = stopAngle - startAngle;

        const auto templateLevel = static_cast<int>(model->layers.size() - 1);
        if (level < 0 || level > templateLevel) {
            // level must greater than 1
            level = templateLevel;
        }
    }

    // TODO copyMakeBorder to enable part match

    std::vector<cv::Mat> pyramids;
    cv::buildPyramid(dst, pyramids, level);

    // match top
    auto candidates = matchTopLevel(pyramids.back(), model->layers[ level ], model->startAngle,
                                    startAngle, spanAngle, maxOverlap, minScore, maxCount, level);

    // match candidate each level
    std::vector<Candidate> levelMatched =
        matchDownLevel(pyramids, model->startAngle, candidates, minScore, subpixel, model, level);

    if (levelMatched.empty()) {
        return {};
    }

    return {};
}

std::vector<Candidate> matchTopLevel(const cv::Mat &dstTop, double startAngle, double spanAngle,
                                     double maxOverlap, double minScore, int maxCount,
                                     const Model *model, int level) {
    std::vector<Candidate> candidates;

    const auto &templateTop       = model->pyramids[ level ];
    auto        angleStep         = sizeAngleStep(templateTop.size());
    cv::Point2d center            = sizeCenter(dstTop.size());
    const auto  topScoreThreshold = minScore * pow(0.9, level);
    bool calMaxByBlock = (dstTop.size().area() / templateTop.size().area() > 500) && maxCount > 10;

    const auto count = static_cast<int>(spanAngle / angleStep) + 1;
    for (int i = 0; i < count; i++) {
        const auto angle  = startAngle + angleStep * i;
        auto       rotate = cv::getRotationMatrix2D(center, angle, 1.);
        auto       size   = computeRotationSize(dstTop.size(), templateTop.size(), angle, rotate);

        auto tx                  = (size.width - 1) / 2. - center.x;
        auto ty                  = (size.height - 1) / 2. - center.y;
        rotate.at<double>(0, 2) += tx;
        rotate.at<double>(1, 2) += ty;
        cv::Point2d offset(tx, ty);

        cv::Mat rotated;
        cv::warpAffine(dstTop, rotated, rotate, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                       model->borderColor);

        cv::Mat result;
        matchTemplate(rotated, result, model, level);
        if (calMaxByBlock) {
            BlockMax  block(result, templateTop.size());
            double    maxScore;
            cv::Point maxPos;
            block.maxValueLoc(maxScore, maxPos);
            if (maxScore < topScoreThreshold) {
                continue;
            }

            candidates.emplace_back(cv::Point2d(maxPos) - offset, angle, maxScore);
            for (int j = 0; j < maxCount + CANDIDATE - 1; j++) {
                nextMaxLoc(maxPos, templateTop.size(), maxOverlap, block, maxScore, maxPos);
                if (maxScore < topScoreThreshold) {
                    break;
                }

                candidates.emplace_back(cv::Point2d(maxPos) - offset, angle, maxScore);
            }
        } else {
            double    maxScore;
            cv::Point maxPos;
            cv::minMaxLoc(result, nullptr, &maxScore, nullptr, &maxPos);
            if (maxScore < topScoreThreshold) {
                continue;
            }

            candidates.emplace_back(cv::Point2d(maxPos) - offset, angle, maxScore);
            for (int j = 0; j < maxCount + CANDIDATE - 1; j++) {
                nextMaxLoc(result, maxPos, templateTop.size(), maxOverlap, maxScore, maxPos);
                if (maxScore < topScoreThreshold) {
                    break;
                }

                candidates.emplace_back(cv::Point2d(maxPos) - offset, angle, maxScore);
            }
        }
    }

    std::sort(candidates.begin(), candidates.end());

    return candidates;
}

std::vector<Candidate> matchDownLevel(const std::vector<cv::Mat>   &pyramids,
                                      const std::vector<Candidate> &candidates, double minScore,
                                      int subpixel, const Model *model, int level) {
    std::vector<Candidate> levelMatched;

    for (const auto &candidate : candidates) {
        auto pose    = candidate;
        bool matched = true;
        for (int currentLevel = level - 1; currentLevel >= 0; currentLevel--) {
            const auto &currentTemplateImg = model->pyramids[ currentLevel ];
            const auto  tmpSize            = currentTemplateImg.size();

            const auto &currentDstImg = pyramids[ currentLevel ];
            const auto  dstSize       = currentDstImg.size();

            auto currentAngleStep = sizeAngleStep(tmpSize);
            auto center           = sizeCenter(dstSize);

            const auto lastSize   = pyramids[ currentLevel + 1 ].size();
            auto       lastCenter = sizeCenter(lastSize);
            auto       topLeft    = transform(pose.pos, lastCenter, -pose.angle) * 2;

            const auto scoreThreshold = minScore * pow(0.9, currentLevel);

            Candidate newCandidate;
            cv::Mat   newScoreRect;
            for (int i = -1; i <= 1; i++) {
                auto angle  = pose.angle + i * currentAngleStep;
                auto rotate = cv::getRotationMatrix2D(center, angle, 1.);

                cv::Mat roi;
                cropRotatedRoi(currentDstImg, tmpSize, topLeft, rotate, roi);

                cv::Mat result;
                matchTemplate(roi, result, model, currentLevel);

                double    maxScore;
                cv::Point maxPos;
                cv::minMaxLoc(result, nullptr, &maxScore, nullptr, &maxPos);

                if (newCandidate.score >= maxScore || maxScore < scoreThreshold) {
                    continue;
                }

                newCandidate = {cv::Point2d(maxPos), angle, maxScore};
                if (0 == currentLevel && 1 == subpixel) {
                    auto isBorder = 0 == maxPos.x || 0 == maxPos.y || result.cols - 1 == maxPos.x ||
                                    result.rows - 1 == maxPos.y;
                    newScoreRect = isBorder
                                       ? cv::Mat()
                                       : result(cv::Rect(maxPos - cv::Point{1, 1}, cv::Size(3, 3)));
                }
            }

            if (newCandidate.score < scoreThreshold) {
                matched = false;
                break;
            }

            if (!newScoreRect.empty()) {
                // TODO subpixel
            }

            // back to
            auto paddingTopLeft =
                transform(topLeft, center, newCandidate.angle) - cv::Point2d(3, 3);
            newCandidate.pos += paddingTopLeft;

            pose = newCandidate;
        }

        if (!matched) {
            continue;
        }

        const auto lastSize   = pyramids.front().size();
        auto       lastCenter = sizeCenter(lastSize);
        pose.pos              = transform(pose.pos, lastCenter, -pose.angle);

        levelMatched.push_back(pose);
    }
    std::sort(levelMatched.begin(), levelMatched.end());

    return levelMatched;
}

std::vector<Pose> matchModel(const cv::Mat &dst, const Model *model, int level,
                             const double startAngle, const double spanAngle,
                             const double maxOverlap, const double minScore, const int maxCount,
                             const int subpixel) {
    // prepare
    {
        if (dst.empty() || nullptr == model) {
            return {};
        }

        auto &templateImg = model->pyramids.front();
        if (dst.cols < templateImg.cols || dst.rows < templateImg.rows ||
            dst.size().area() < templateImg.size().area()) {
            return {};
        }

        const auto templateLevel = static_cast<int>(model->pyramids.size() - 1);
        if (level < 0 || level > templateLevel) {
            // level must greater than 1
            level = templateLevel;
        }
    }

    std::vector<cv::Mat> pyramids;
    cv::buildPyramid(dst, pyramids, level);

    // compute top
    const std::vector<Candidate> candidates = matchTopLevel(
        pyramids.back(), startAngle, spanAngle, maxOverlap, minScore, maxCount, model, level);

    // match candidate each level
    std::vector<Candidate> levelMatched =
        matchDownLevel(pyramids, candidates, minScore, subpixel, model, level);

    // filter overlap
    std::vector<cv::RotatedRect> rects;
    {
        rects.reserve(levelMatched.size());
        const auto        size = model->pyramids.front().size();
        const cv::Point2f topRight(static_cast<float>(size.width), 0.f);
        const cv::Point2f bottomRight(static_cast<float>(size.width),
                                      static_cast<float>(size.height));
        for (const auto &candidate : levelMatched) {
            std::vector<cv::Point2f> points{topRight + cv::Point2f(candidate.pos),
                                            bottomRight + cv::Point2f(candidate.pos)};
            auto rotate = cv::getRotationMatrix2D(candidate.pos, -candidate.angle, 1.);
            std::vector<cv::Point2f> rotatedPoints;
            cv::transform(points, rotatedPoints, rotate);

            rects.emplace_back(cv::Point2f(candidate.pos), rotatedPoints[ 0 ], rotatedPoints[ 1 ]);
        }
        filterOverlap(levelMatched, rects, maxOverlap);
    }

    std::vector<Pose> result;
    {
        const auto count = levelMatched.size();
        for (std::size_t i = 0; i < count; i++) {
            const auto &candidate = levelMatched[ i ];
            const auto &rect      = rects[ i ];

            if (candidate.score < 0) {
                continue;
            }

            const auto &center = rect.center;
            result.emplace_back(Pose{center.x, center.y, static_cast<float>(-candidate.angle),
                                     static_cast<float>(candidate.score)});
        }

        std::sort(result.begin(), result.end(),
                  [](const Pose &a, const Pose &b) { return a.score > b.score; });
    }

    return result;
}
