#include "grayMatch.h"
#include "privateType.h"

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/opencv.hpp>
#include <utility>

constexpr int    MIN_AREA  = 256;
constexpr double TOLERANCE = 0.0000001;
constexpr int    CANDIDATE = 5;
constexpr double INVALID   = -1.;

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

inline cv::Point2d transform(const cv::Point2d &point, const cv::Mat &rotate) {
    const auto ptr = rotate.ptr<double>();

    auto x = point.x * ptr[ 0 ] + point.y * ptr[ 1 ] + ptr[ 2 ];
    auto y = point.x * ptr[ 3 ] + point.y * ptr[ 4 ] + ptr[ 5 ];

    return {x, y};
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

    const std::vector<cv::Point2d> pt{
        transform({0, 0}, rotate), transform({static_cast<double>(dstSize.width) - 1, 0}, rotate),
        transform({0, static_cast<double>(dstSize.height) - 1}, rotate),
        transform({static_cast<double>(dstSize.width) - 1, static_cast<double>(dstSize.height) - 1},
                  rotate)};

    cv::Point2d min;
    cv::Point2d max;
    min.x = std::min(std::min(std::min(pt[ 0 ].x, pt[ 1 ].x), pt[ 2 ].x), pt[ 3 ].x);
    min.y = std::min(std::min(std::min(pt[ 0 ].y, pt[ 1 ].y), pt[ 2 ].y), pt[ 3 ].y);
    max.x = std::max(std::max(std::max(pt[ 0 ].x, pt[ 1 ].x), pt[ 2 ].x), pt[ 3 ].x);
    max.y = std::max(std::max(std::max(pt[ 0 ].y, pt[ 1 ].y), pt[ 2 ].y), pt[ 3 ].y);

    if (angle > 0 && angle < 90) {
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

    const auto center     = cv::Point2d((dstSize.width - 1.) / 2., (dstSize.height - 1.) / 2.);
    const auto halfHeight = static_cast<int>(ceil(max.y - center.y - width));
    const auto halfWidth  = static_cast<int>(ceil(max.x - center.x - height));

    cv::Size   size(halfWidth * 2, halfHeight * 2);
    const auto wrongSize = (templateSize.width < size.width && templateSize.height > size.height) ||
                           (templateSize.width > size.width && templateSize.height < size.height) ||
                           templateSize.area() > size.area();
    if (wrongSize) {
        size = {static_cast<int>(lround(max.x - min.x + 0.5)),
                static_cast<int>(lround(max.y - min.y + 0.5))};
    }

    return size;
}

#ifdef CV_SIMD
#include "integral.h"

void matchTemplateSimd(const cv::Mat &src, const cv::Mat &templateImg, cv::Mat &result) {
    result.create(src.size() - templateImg.size() + cv::Size(1, 1), CV_32FC1);

    auto       *resultStart = result.ptr<float>();
    const auto  resultStep  = result.step[ 0 ] / result.step[ 1 ];
    const auto *srcStart    = src.ptr<uchar>();
    const auto  srcStep     = src.step[ 0 ];
    const auto *temStart    = templateImg.ptr<uchar>();
    const auto  temStep     = templateImg.step[ 0 ];

    std::vector<cv::v_uint32> tmp(result.cols, cv::v_setzero_u32());
    for (int y = 0; y < result.rows; y++) {
        auto *resultPtr = resultStart + resultStep * y;

        for (int templateRow = 0; templateRow < templateImg.rows; templateRow++) {
            auto *temPtr = temStart + temStep * templateRow;

            for (int i = 0; i < templateImg.cols; i += simdSize(cv::v_uint8)) {
                auto  vTem   = cv::v_load_aligned(temPtr + i);
                auto *srcPtr = srcStart + srcStep * (y + templateRow) + i;

                int x = 0;
                for (int n = 0; n < result.cols / 4; n++) {
                    auto vSrc = cv::v_load(srcPtr + x);
                    tmp[ x ]  = cv::v_add(tmp[ x ], cv::v_dotprod_expand_fast(vSrc, vTem));
                    x++;

                    vSrc     = cv::v_load(srcPtr + x);
                    tmp[ x ] = cv::v_add(tmp[ x ], cv::v_dotprod_expand_fast(vSrc, vTem));
                    x++;

                    vSrc     = cv::v_load(srcPtr + x);
                    tmp[ x ] = cv::v_add(tmp[ x ], cv::v_dotprod_expand_fast(vSrc, vTem));
                    x++;

                    vSrc     = cv::v_load(srcPtr + x);
                    tmp[ x ] = cv::v_add(tmp[ x ], cv::v_dotprod_expand_fast(vSrc, vTem));
                    x++;
                }
                for (; x < result.cols; x++) {
                    auto vSrc = cv::v_load(srcPtr + x);
                    tmp[ x ]  = cv::v_add(tmp[ x ], cv::v_dotprod_expand_fast(vSrc, vTem));
                }
            }
        }

        for (int x = 0; x < result.cols; x++) {
            const auto sum = cv::v_reduce_sum(tmp[ x ]);
            tmp[ x ]       = cv::v_setzero_u32();
            resultPtr[ x ] = static_cast<float>(sum);
        }
    }
}
#endif

void ccoeffDenominator(const cv::Mat &src, const cv::Size &templateSize, cv::Mat &result,
                       const double mean, const double normal, const double invArea,
                       const bool equal1) {
    if (equal1) {
        result = cv::Scalar::all(1);
        return;
    }

    cv::Mat sum;
    cv::Mat sqSum;
#ifdef CV_SIMD
    integralSimd(src, sum, sqSum);
#else
    cv::integral(src, sum, sqSum, CV_64F);
#endif

    const auto *q0 = reinterpret_cast<double *>(sqSum.data);
    const auto *q1 = q0 + templateSize.width;
    const auto *q2 = reinterpret_cast<double *>(sqSum.data) + sqSum.step1() * templateSize.height;
    const auto *q3 = q2 + templateSize.width;

    const auto *p0 = reinterpret_cast<double *>(sum.data);
    const auto *p1 = p0 + templateSize.width;
    const auto *p2 = reinterpret_cast<double *>(sum.data) + sum.step1() * templateSize.height;
    const auto *p3 = p2 + templateSize.width;

    const auto step           = sum.step / sizeof(double);
    auto      *resultStartPtr = reinterpret_cast<float *>(result.data);

    for (int y = 0; y < result.rows; y++) {
        auto *scorePtr = resultStartPtr + y * result.step1();
        auto  idx      = y * step;
        for (int x = 0; x < result.cols; x++, idx++) {
            auto      &score     = scorePtr[ x ];
            const auto partSum   = p0[ idx ] - p1[ idx ] - p2[ idx ] + p3[ idx ];
            const auto numerator = score - partSum * mean;

            const auto partSqSum    = q0[ idx ] - q1[ idx ] - q2[ idx ] + q3[ idx ];
            auto       partSqNormal = partSqSum - partSum * partSum * invArea;

            const auto   diff = std::max(partSqNormal, 0.);
            const double denominator =
                diff <= std::min(0.5, 10 * FLT_EPSILON * partSqSum) ? 0 : sqrt(diff) * normal;

            if (abs(numerator) < denominator) {
                score = static_cast<float>(numerator / denominator);
            } else if (abs(numerator) < denominator * 1.125) {
                score = numerator > 0.f ? 1.f : -1.f;
            } else {
                score = 0;
            }
        }
    }
}

void matchTemplate(cv::Mat &src, cv::Mat &result, const Model *model, int level) {
#ifdef CV_SIMD
    matchTemplateSimd(src, model->pyramids[ level ], result);
#else
    cv::matchTemplate(src, model->pyramids[ level ], result, cv::TM_CCORR);
#endif
    ccoeffDenominator(src, model->pyramids[ level ].size(), result, model->mean[ level ][ 0 ],
                      model->normal[ level ], model->invArea[ level ], model->equal1[ level ] == 1);
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

inline cv::Mat getRotationMatrix2D(const cv::Point2f &center, double angle) {
    angle              *= CV_PI / 180;
    const double alpha  = std::cos(angle);
    const double beta   = std::sin(angle);

    cv::Mat    rotate(2, 3, CV_64FC1);
    const auto ptr = rotate.ptr<double>();
    ptr[ 0 ]       = alpha;
    ptr[ 1 ]       = beta;
    ptr[ 2 ]       = (1 - alpha) * center.x - beta * center.y;
    ptr[ 3 ]       = -beta;
    ptr[ 4 ]       = alpha;
    ptr[ 5 ]       = beta * center.x + (1 - alpha) * center.y;

    return rotate;
}

inline cv::Point2d transform(const cv::Point2d &point, const cv::Point &center,
                             const double angle) {
    const auto rotate = getRotationMatrix2D(center, angle);

    return transform(point, rotate);
}

inline cv::Point2d sizeCenter(const cv::Size &size) {
    return {(size.width - 1.) / 2., (size.height - 1.) / 2.};
}

void cropRotatedRoi(const cv::Mat &src, const cv::Size &templateSize, const cv::Point2d topLeft,
                    cv::Mat &rotate, cv::Mat &roi) {
    const auto     point = transform(topLeft, rotate);
    const cv::Size paddingSize(templateSize.width + 6, templateSize.height + 6);
    rotate.at<double>(0, 2) -= point.x - 3;
    rotate.at<double>(1, 2) -= point.y - 3;

    cv::warpAffine(src, roi, rotate, paddingSize);
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

                    const auto area = cv::contourArea(points);
                    const auto overlap =
                        area / static_cast<double>(rect.size.width * rect.size.height);
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

Model *trainModel(const cv::Mat &src, int level) {
    if (src.empty() || src.channels() != 1) {
        return nullptr;
    }

    if (level <= 0) {
        // level must greater than 0
        level = computeLayers(src.size().width, src.size().height, MIN_AREA);
    }

    const auto scale   = 1 << (level - 1);
    const auto topArea = src.size().area() / (scale * scale);
    if (MIN_AREA > topArea) {
        // top area must greater than MIN_AREA
        return nullptr;
    }

    auto  *result = new Model;
    Model &model  = *result;

    std::vector<cv::Mat> pyramids;
    cv::buildPyramid(src, pyramids, level);
    model.borderColor = cv::mean(src).val[ 0 ] < 128 ? 255 : 0;
    model.reserve(pyramids.size());

    for (const auto &pyramid : pyramids) {
        int     alignedWidth = static_cast<int>(cv::alignSize(pyramid.cols, simdSize(cv::v_uint8)));
        auto    img          = cv::Mat::zeros(pyramid.rows, alignedWidth, CV_8UC1);
        cv::Mat sub          = img(cv::Rect(0, 0, pyramid.cols, pyramid.rows));
        for (int y = 0; y < pyramid.rows; y++) {
            auto *dstPtr = sub.ptr<uchar>(y);
            auto *srcPtr = pyramid.ptr<uchar>(y);
            memcpy(dstPtr, srcPtr, pyramid.cols);
        }

        model.pyramids.push_back(sub);

        auto invArea = 1. / pyramid.size().area();

        cv::Scalar mean;
        cv::Scalar stdDev;
        cv::meanStdDev(pyramid, mean, stdDev);

        const auto stdNormal = stdDev[ 0 ] * stdDev[ 0 ] + stdDev[ 1 ] * stdDev[ 1 ] +
                               stdDev[ 2 ] * stdDev[ 2 ] + stdDev[ 3 ] * stdDev[ 3 ];
        auto equal1 = stdNormal < std::numeric_limits<double>::epsilon();
        auto normal = sqrt(stdNormal) / sqrt(invArea);

        model.equal1.push_back(equal1);
        model.invArea.push_back(invArea);
        model.mean.push_back(mean);
        model.normal.push_back(normal);
    }

    return result;
}

inline double sizeAngleStep(const cv::Size &size) {
    return atan(2. / std::max(size.width, size.height)) * 180. / CV_PI;
}

#pragma omp declare reduction(combine : std::vector<Candidate> : omp_out.insert(                   \
        omp_out.end(), omp_in.begin(), omp_in.end()))

std::vector<Candidate> matchTopLevel(const cv::Mat &dstTop, double startAngle, double spanAngle,
                                     double maxOverlap, double minScore, int maxCount,
                                     const Model *model, int level) {
    std::vector<Candidate> candidates;

    const auto &templateTop       = model->pyramids[ level ];
    auto        angleStep         = sizeAngleStep(templateTop.size());
    cv::Point2d center            = sizeCenter(dstTop.size());
    const auto  topScoreThreshold = minScore * pow(0.9, level);
    bool calMaxByBlock = dstTop.size().area() / templateTop.size().area() > 500 && maxCount > 10;

    const auto count = static_cast<int>(spanAngle / angleStep) + 1;
#pragma omp parallel for reduction(combine : candidates)
    for (int i = 0; i < count; i++) {
        const auto angle  = startAngle + angleStep * i;
        auto       rotate = getRotationMatrix2D(center, angle);
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

cv::Point2f computeSubpixel(const cv::Mat &score) {
    cv::Point2f result(0, 0);
    auto       *mag = score.ptr<float>();

    const auto gx  = (-mag[ 0 ] + mag[ 2 ] - mag[ 3 ] + mag[ 5 ] - mag[ 6 ] + mag[ 8 ]) / 3.0f;
    const auto gy  = (mag[ 6 ] + mag[ 7 ] + mag[ 8 ] - mag[ 0 ] - mag[ 1 ] - mag[ 2 ]) / 3.0f;
    const auto gxx = (mag[ 0 ] - 2.0f * mag[ 1 ] + mag[ 2 ] + mag[ 3 ] - 2.0f * mag[ 4 ] +
                      mag[ 5 ] + mag[ 6 ] - 2.0f * mag[ 7 ] + mag[ 8 ]) /
                     6.0f;
    const auto gxy = (-mag[ 0 ] + mag[ 2 ] + mag[ 6 ] - mag[ 8 ]) / 2.0f;
    const auto gyy = (mag[ 0 ] + mag[ 1 ] + mag[ 2 ] - 2.0f * (mag[ 3 ] + mag[ 4 ] + mag[ 5 ]) +
                      mag[ 6 ] + mag[ 7 ] + mag[ 8 ]) /
                     6.0f;

    cv::Mat hessian(2, 2, CV_32FC1);
    hessian.at<float>(0, 0) = gxx;
    hessian.at<float>(0, 1) = gxy;
    hessian.at<float>(1, 0) = gxy;
    hessian.at<float>(1, 1) = gyy;

    cv::Mat eigenvalue;
    cv::Mat eigenvector;
    cv::eigen(hessian, eigenvalue, eigenvector);
    float nx, ny;
    if (fabs(eigenvalue.at<float>(0, 0)) >= fabs(eigenvalue.at<float>(1, 0))) {
        nx = eigenvector.at<float>(0, 0);
        ny = eigenvector.at<float>(0, 1);
    } else {
        nx = eigenvector.at<float>(1, 0);
        ny = eigenvector.at<float>(1, 1);
    }

    const auto denominator = gxx * nx * nx + 2 * gxy * nx * ny + gyy * ny * ny;
    if (denominator != 0.0) {
        const auto t = -(gx * nx + gy * ny) / denominator;
        result.x     = t * nx;
        result.y     = t * ny;
    }

    return result;
}

std::vector<Candidate> matchDownLevel(const std::vector<cv::Mat>   &pyramids,
                                      const std::vector<Candidate> &candidates, double minScore,
                                      int subpixel, const Model *model, int level) {
    std::vector<Candidate> levelMatched;
    auto                   count = static_cast<int>(candidates.size());

#pragma omp parallel for reduction(combine : levelMatched) schedule(dynamic)
    for (int index = 0; index < count; index++) {
        auto pose    = candidates[ index ];
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
                auto rotate = getRotationMatrix2D(center, angle);

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
                auto offset         = computeSubpixel(newScoreRect);
                newCandidate.pos.x += offset.x;
                newCandidate.pos.y += offset.y;
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
            auto                     rotate = getRotationMatrix2D(candidate.pos, -candidate.angle);
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

Model_t trainModel(const unsigned char *data, int width, int height, int channels, int bytesPerLine,
                   int roiLeft, int roiTop, int roiWidth, int roiHeight, int levelNum) {
    if ((1 != channels && 3 != channels && 4 != channels) || nullptr == data) {
        return nullptr;
    }

    const auto    type = channels == 1 ? CV_8UC1 : channels == 3 ? CV_8UC3 : CV_8UC4;
    const cv::Mat img(cv::Size(width, height), type, const_cast<unsigned char *>(data),
                      bytesPerLine);

    cv::Mat src;
    if (1 == channels) {
        src = img;
    } else {
        cv::cvtColor(img, src, channels == 3 ? cv::COLOR_RGB2GRAY : cv::COLOR_RGBA2GRAY);
    }

    const cv::Rect rect(roiLeft, roiTop, roiWidth, roiHeight);
    const cv::Rect imageRect(0, 0, width, height);
    const auto     roi = rect & imageRect;
    if (roi.empty()) {
        return nullptr;
    }

    return trainModel(src(roi), levelNum);
}

void matchModel(const unsigned char *data, int width, int height, int channels, int bytesPerLine,
                int roiLeft, int roiTop, int roiWidth, int roiHeight, Model *const model,
                int *count, Pose *poses, int level, double startAngle, double spanAngle,
                double maxOverlap, double minScore, int subpixel) {
    if (nullptr == count) {
        return;
    }

    if (nullptr == poses || nullptr == data) {
        *count = 0;
        return;
    }

    if (1 != channels && 3 != channels && 4 != channels) {
        *count = 0;
        return;
    }

    const auto    type = channels == 1 ? CV_8UC1 : channels == 3 ? CV_8UC3 : CV_8UC4;
    const cv::Mat img(cv::Size(width, height), type, const_cast<unsigned char *>(data),
                      bytesPerLine);

    cv::Mat dst;
    if (1 == channels) {
        dst = img;
    } else {
        cv::cvtColor(img, dst, channels == 3 ? cv::COLOR_RGB2GRAY : cv::COLOR_RGBA2GRAY);
    }

    const cv::Rect rect(roiLeft, roiTop, roiWidth, roiHeight);
    const cv::Rect imageRect(0, 0, width, height);
    const auto     roi = rect & imageRect;
    if (roi.empty()) {
        *count = 0;
        return;
    }

    const auto result = matchModel(dst(roi), model, level, startAngle, spanAngle, maxOverlap,
                                   minScore, *count, subpixel);

    const auto size = std::min(*count, static_cast<int>(result.size()));
    for (int i = 0; i < size; i++) {
        const auto &pose = result[ i ];
        poses[ i ]       = {pose.x + static_cast<float>(roi.x), pose.y + static_cast<float>(roi.y),
                            pose.angle, pose.score};
    }

    *count = size;
}

void freeModel(Model_t *model) {
    if (nullptr == model || nullptr == *model) {
        return;
    }

    delete *model;
    *model = nullptr;
}

int modelLevel(Model *const model) {
    if (nullptr == model) {
        return 0;
    }

    return static_cast<int>(model->pyramids.size());
}

void modelImage(Model *const model, int level, unsigned char *data, int length, int *width,
                int *height, int *channels) {
    if (nullptr == model) {
        return;
    }

    if (level < 0 || level > static_cast<int>(model->pyramids.size() - 1)) {
        return;
    }

    const auto &img = model->pyramids[ level ];
    if (nullptr != width) {
        *width = img.cols;
    }
    if (nullptr != height) {
        *height = img.rows;
    }
    if (nullptr != channels) {
        *channels = img.channels();
    }

    if (nullptr == data || length < img.cols * img.rows * img.channels()) {
        return;
    }

    for (int y = 0; y < img.rows; y++) {
        const auto *ptr = img.ptr<unsigned char>(y);
        memcpy(data + y * img.cols, ptr, img.cols);
    }
}
