#include "grayMatch.h"
#include "privateType.h"
#include "sum.h"

#include <opencv2/core/hal/intrin.hpp>

constexpr int    MIN_AREA  = 256;
constexpr double TOLERANCE = 0.0000001;
constexpr int    CANDIDATE = 5;
constexpr double INVALID   = -1.;

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

inline void RotatedRectPoints(const cv::RotatedRect &rect, cv::Point2f pt[]) {
    const auto angle = -rect.angle * CV_PI / 180.;
    const auto b     = static_cast<float>(cos(angle)) * 0.5f;
    const auto a     = static_cast<float>(sin(angle)) * 0.5f;

    pt[ 0 ].x = rect.center.x - a * rect.size.height - b * rect.size.width;
    pt[ 0 ].y = rect.center.y + b * rect.size.height - a * rect.size.width;
    pt[ 1 ].x = rect.center.x + a * rect.size.height - b * rect.size.width;
    pt[ 1 ].y = rect.center.y - b * rect.size.height - a * rect.size.width;
    pt[ 2 ].x = 2 * rect.center.x - pt[ 0 ].x;
    pt[ 2 ].y = 2 * rect.center.y - pt[ 0 ].y;
    pt[ 3 ].x = 2 * rect.center.x - pt[ 1 ].x;
    pt[ 3 ].y = 2 * rect.center.y - pt[ 1 ].y;
}

inline void RotatedRectPoints(const cv::RotatedRect &rect, std::vector<cv::Point2f> &pts) {
    pts.resize(4);
    RotatedRectPoints(rect, pts.data());
}

inline cv::Rect boundingRect(const cv::RotatedRect &rect) {
    auto angle = rect.angle;
    if (angle > 360) {
        angle -= 360;
    } else if (angle < 0) {
        angle += 360;
    }

    if (fabs(fabs(angle) - 90) < TOLERANCE || fabs(fabs(angle) - 270) < TOLERANCE) {
        const auto halfWidth  = rect.size.height / 2.;
        const auto halfHeight = rect.size.width / 2;
        const auto x          = rect.center.x - halfWidth;
        const auto y          = rect.center.y - halfHeight;
        return {cvFloor(x), cvFloor(y), cvCeil(rect.size.height), cvCeil(rect.size.width)};
    }

    if (fabs(angle) < TOLERANCE || fabs(fabs(angle) - 180) < TOLERANCE) {
        const auto halfWidth  = rect.size.width / 2.;
        const auto halfHeight = rect.size.height / 2;
        const auto x          = rect.center.x - halfWidth;
        const auto y          = rect.center.y - halfHeight;
        return {cvFloor(x), cvFloor(y), cvCeil(rect.size.width), cvCeil(rect.size.height)};
    }

    cv::Point2f pt[ 4 ];
    RotatedRectPoints(rect, pt);
    cv::Rect r(cvFloor(std::min(std::min(std::min(pt[ 0 ].x, pt[ 1 ].x), pt[ 2 ].x), pt[ 3 ].x)),
               cvFloor(std::min(std::min(std::min(pt[ 0 ].y, pt[ 1 ].y), pt[ 2 ].y), pt[ 3 ].y)),
               cvCeil(std::max(std::max(std::max(pt[ 0 ].x, pt[ 1 ].x), pt[ 2 ].x), pt[ 3 ].x)),
               cvCeil(std::max(std::max(std::max(pt[ 0 ].y, pt[ 1 ].y), pt[ 2 ].y), pt[ 3 ].y)));
    r.width  -= r.x - 1;
    r.height -= r.y - 1;
    return r;
}

inline cv::Point2d transform(const cv::Point2d &point, const cv::Mat &rotate) {
    const auto ptr = rotate.ptr<double>();

    auto x = point.x * ptr[ 0 ] + point.y * ptr[ 1 ] + ptr[ 2 ];
    auto y = point.x * ptr[ 3 ] + point.y * ptr[ 4 ] + ptr[ 5 ];

    return {x, y};
}

inline cv::Point2d sizeCenter(const cv::Size &size) {
    return {(size.width - 1.) / 2., (size.height - 1.) / 2.};
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

#ifdef CV_SIMD
void matchTemplateSimd(const cv::Mat &src, const cv::Mat &templateImg, const HRegion &hRegion,
                       cv::Mat &result) {
    result.create(src.size() - templateImg.size() + cv::Size(1, 1), CV_32FC1);

    auto       *resultStart = (float *)result.data;
    const auto  resultStep  = result.step1();
    const auto *srcStart    = src.data;
    const auto  srcStep     = src.step1();
    const auto *temStart    = templateImg.data;
    const auto  temStep     = templateImg.step1();

    std::vector<cv::v_uint32> tmp(result.cols, cv::v_setzero_u32());
    for (int y = 0; y < result.rows; y++) {
        auto *resultPtr = resultStart + resultStep * y;

        for (const auto &rle : hRegion) {
            auto *temPtr  = temStart + temStep * rle.row + rle.startColumn;
            auto *srcPtr0 = srcStart + srcStep * (y + rle.row) + rle.startColumn;

            for (int i = 0; i < rle.length; i += simdSize(cv::v_uint8)) {
                auto  vTem   = cv::v_load(temPtr + i);
                auto *srcPtr = srcPtr0 + i;

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
            auto sum       = cv::v_reduce_sum(tmp[ x ]);
            tmp[ x ]       = cv::v_setzero_u32();
            resultPtr[ x ] = static_cast<float>(sum);
        }
    }
}

void ccoeffDenominator(const cv::Mat &src, const cv::Size &tempSize, const HRegion &hRegion,
                       const VRegion &vRegion, cv::Mat &result, const double mean,
                       const double normal, const double invArea) {
    cv::Mat sumImg;
    cv::Mat sqSumImg;
    integralSum(src, sumImg, sqSumImg, tempSize, hRegion, vRegion);

    auto      *resultStart   = (float *)result.data;
    const auto resultStep    = result.step1();
    auto      *sumStartPtr   = (double *)sumImg.data;
    const auto sumStep       = sumImg.step1();
    auto      *sqSumStartPtr = (double *)sqSumImg.data;
    const auto sqSumStep     = sqSumImg.step1();

    for (int y = 0; y < result.rows; y++) {
        auto *resultPtr = resultStart + resultStep * y;
        auto *sumPtr    = sumStartPtr + sumStep * y;
        auto *sqSumPtr  = sqSumStartPtr + sqSumStep * y;
        for (int x = 0; x < result.cols; x++) {
            auto &score = resultPtr[ x ];
            auto  sum   = sumPtr[ x ];
            auto  sqSum = sqSumPtr[ x ];

            const auto numerator    = static_cast<double>(score) - sum * mean;
            const auto partSqNormal = sqSum - static_cast<double>(sum * sum) * invArea;
            const auto diff         = std::max(partSqNormal, 0.);
            const auto denominator =
                (diff <= std::min(0.5, 10 * FLT_EPSILON * sqSum)) ? 0 : sqrt(diff) * normal;

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
#endif

void matchTemplate(const cv::Mat &src, cv::Mat &result, const Template &layerTemplate) {
#ifdef CV_SIMD
    matchTemplateSimd(src, layerTemplate.img, layerTemplate.hRegion, result);
    ccoeffDenominator(src, layerTemplate.img.size(), layerTemplate.hRegion, layerTemplate.vRegion,
                      result, layerTemplate.mean, layerTemplate.normal, layerTemplate.invArea);

#else
    // make mask at train model process
    cv::Point2f pts[ 4 ];
    RotatedRectPoints(layerTemplate.rect, pts);
    cv::Mat roi = cv::Mat::zeros(layerTemplate.img.size(), CV_8UC1);
    cv::fillConvexPoly(roi,
                       std::vector<cv::Point>{cv::Point(pts[ 0 ]), cv::Point(pts[ 1 ]),
                                              cv::Point(pts[ 2 ]), cv::Point(pts[ 3 ])},
                       cv::Scalar(255));

    cv::matchTemplate(src, layerTemplate.img, result, cv::TM_CCOEFF_NORMED, roi);

#endif // CV_SIMD
}

void nextMaxLoc(cv::Mat &score, const cv::Point &pos, const cv::RotatedRect &rect,
                const double maxOverlap, double &maxScore, cv::Point &maxPos) {
    const auto alone = static_cast<float>(1.f - maxOverlap);

    auto rectIgnore         = rect;
    rectIgnore.center       = cv::Point2f(pos);
    rectIgnore.size.width  *= (2 * alone);
    rectIgnore.size.height *= (2 * alone);

    // clear neighbor
    std::vector<cv::Point2f> pts;
    RotatedRectPoints(rectIgnore, pts);
    cv::fillConvexPoly(score,
                       std::vector<cv::Point>{cv::Point(pts[ 0 ]), cv::Point(pts[ 1 ]),
                                              cv::Point(pts[ 2 ]), cv::Point(pts[ 3 ])},
                       cv::Scalar(0));

    cv::minMaxLoc(score, nullptr, &maxScore, nullptr, &maxPos);
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

HRegion horizonRegion(const cv::Mat &src) {
    HRegion result;

    for (int y = 0; y < src.rows; y++) {
        auto *ptr    = src.ptr<uchar>(y);
        bool  inside = false;
        HRLE  rle;
        for (int x = 0; x < src.cols; x++) {
            if (0 == ptr[ x ]) {
                if (inside) {
                    break;
                }

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
        }

        if (inside) {
            result.push_back(rle);
        }
    }

    return result;
}

VRegion verticalRegion(const cv::Mat &src) {
    VRegion result;

    for (int x = 0; x < src.cols; x++) {
        auto *ptr    = src.ptr<uchar>() + x;
        bool  inside = false;
        VRLE  rle;
        for (int y = 0; y < src.rows; y++) {
            if (0 == ptr[ y * src.step ]) {
                if (inside) {
                    break;
                }

                continue;
            }

            if (!inside) {
                inside       = true;
                rle.col      = x;
                rle.startRow = y;
                rle.length   = 1;
                continue;
            }

            rle.length++;
        }

        if (inside) {
            result.push_back(rle);
        }
    }

    return result;
}

int regionArea(const HRegion &region) {
    int area = 0;
    for (const auto &rle : region) {
        area += rle.length;
    }

    return area;
}

double normalizeAngle(const double angle) {
    if (angle < 0.0) {
        const int k = static_cast<int>(std::ceil(-angle / 360.0));
        return angle + k * 360.0;
    }

    const int k = static_cast<int>(std::floor(angle / 360.0));
    return angle - k * 360.0;
}

Model *trainModel(const cv::Mat &src, int level, double startAngle, double spanAngle,
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

    auto  *result    = new Model;
    Model &model     = *result;
    model.startAngle = startAngle;
    model.angleStep  = angleStep;
    model.stopAngle  = startAngle + angleStep * nAngle;
    model.srcSize    = src.size();
    model.layers.resize(level);

    std::vector<cv::Mat> pyramids;
    cv::buildPyramid(src, pyramids, level - 1);
    for (int i = 0; i < level; i++) {
        const auto &pyramid = pyramids[ i ];

        Layer layer;
        layer.angleStep = angleStep * (1 << i);

        auto count  = static_cast<int>(ceil(spanAngle / layer.angleStep)) + 1;
        auto center = sizeCenter(pyramid.size());
        layer.templates.resize(count);

#pragma omp parallel for
        for (int j = 0; j < count; j++) {
            Template layerTemplate;

            auto angle         = startAngle + layer.angleStep * j;
            layerTemplate.rect = {cv::Point2f(center), pyramid.size(), static_cast<float>(angle)};
            auto rect          = boundingRect(layerTemplate.rect);
            auto newCenter     = sizeCenter(rect.size());

            auto offset = newCenter - cv::Point2d(layerTemplate.rect.center);
            auto rotate = cv::getRotationMatrix2D(layerTemplate.rect.center, angle, 1.);
            rotate.at<double>(0, 2) += offset.x;
            rotate.at<double>(1, 2) += offset.y;

            auto &rotated = layerTemplate.img;
            auto img = cv::Mat::zeros(rect.height, rect.width + simdSize(cv::v_uint8) - 1, CV_8UC1);
            rotated  = img(cv::Rect(0, 0, rect.width, rect.height));
            cv::warpAffine(pyramid, rotated, rotate, rect.size(), cv::INTER_CUBIC,
                           cv::BORDER_DEFAULT);

            layerTemplate.rect.center = newCenter;

            cv::Point2f pts[ 4 ];
            RotatedRectPoints(layerTemplate.rect, pts);
            cv::Mat roi = cv::Mat::zeros(rect.size(), CV_8UC1);
            cv::fillConvexPoly(roi,
                               std::vector<cv::Point>{cv::Point(pts[ 0 ]), cv::Point(pts[ 1 ]),
                                                      cv::Point(pts[ 2 ]), cv::Point(pts[ 3 ])},
                               cv::Scalar(255));

            // draw rotated rect line, then encode with run-length-encode
            layerTemplate.hRegion = horizonRegion(roi);
            layerTemplate.vRegion = verticalRegion(roi);
            cv::bitwise_and(rotated, roi, rotated);

            cv::Scalar mean;
            cv::Scalar stdDev;
            cv::meanStdDev(rotated, mean, stdDev, roi);
            const auto &stdDevVal = stdDev[ 0 ];

            layerTemplate.invArea = 1. / regionArea(layerTemplate.hRegion);
            layerTemplate.normal  = stdDevVal / sqrt(layerTemplate.invArea);
            layerTemplate.mean    = mean[ 0 ];

            layer.templates[ j ] = std::move(layerTemplate);
        }

        model.layers[ i ] = std::move(layer);
    }

    return result;
}

#pragma omp declare reduction(combine : std::vector<Candidate> : omp_out.insert(                   \
        omp_out.end(), omp_in.begin(), omp_in.end()))

std::vector<Candidate> matchTopLevel(const cv::Mat &dstTop, const Layer &layer,
                                     const double modelAngleStart, double startAngle,
                                     double spanAngle, const double maxOverlap,
                                     const double minScore, const int maxCount, const int level) {
    const int startIndex =
        static_cast<int>(floor((startAngle - modelAngleStart) / layer.angleStep));
    const int  count             = static_cast<int>(ceil(spanAngle / layer.angleStep)) + 1;
    const auto topScoreThreshold = minScore * pow(0.9, level);

    std::vector<Candidate> candidates;
#pragma omp parallel for reduction(combine : candidates)
    for (int i = 0; i < count; i++) {
        const auto &layerTemplate = layer.templates[ startIndex + i ];

        cv::Mat result;
        matchTemplate(dstTop, result, layerTemplate);

        double    maxScore;
        cv::Point maxPos;
        cv::minMaxLoc(result, nullptr, &maxScore, nullptr, &maxPos);
        if (maxScore < topScoreThreshold) {
            continue;
        }

        candidates.emplace_back(cv::Point2d(maxPos) + cv::Point2d(layerTemplate.rect.center),
                                layerTemplate.rect.angle, maxScore);
        for (int j = 0; j < maxCount + CANDIDATE - 1; j++) {

            nextMaxLoc(result, maxPos, layerTemplate.rect, maxOverlap, maxScore, maxPos);
            if (maxScore < topScoreThreshold) {
                break;
            }

            candidates.emplace_back(cv::Point2d(maxPos) + cv::Point2d(layerTemplate.rect.center),
                                    layerTemplate.rect.angle, maxScore);
        }
    }
    std::sort(candidates.begin(), candidates.end());

    return candidates;
}

std::vector<Candidate> matchDownLevel(const std::vector<cv::Mat>   &pyramids,
                                      const double                  modelAngleStart,
                                      const std::vector<Candidate> &candidates,
                                      const double minScore, const int subpixel, const Model *model,
                                      int level) {
    std::vector<Candidate> levelMatched;

#pragma omp parallel for reduction(combine : levelMatched) schedule(dynamic)
    for (std::size_t idx = 0; idx < candidates.size(); idx++) {
        auto pose    = candidates[ idx ];
        bool matched = true;
        for (int currentLevel = level - 1; currentLevel >= 0; currentLevel--) {
            const auto &layer = model->layers[ currentLevel ];
            const int   angleIndex =
                static_cast<int>(round((pose.angle - modelAngleStart) / layer.angleStep));
            const auto &currentDstImg  = pyramids[ currentLevel ];
            const auto  center         = pose.pos * 2.;
            const auto  scoreThreshold = minScore * pow(0.9, currentLevel);

            Candidate newCandidate;
            cv::Mat   newScoreRect;
            for (int i = -1; i <= 1; i++) {
                auto index = angleIndex + i;
                if (index < 0 || index >= static_cast<int>(layer.templates.size())) {
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
                cv::Mat roi = currentDstImg(rect);
                matchTemplate(roi, result, layerTemplate);

                double    maxScore;
                cv::Point maxPos;
                cv::minMaxLoc(result, nullptr, &maxScore, nullptr, &maxPos);

                if (newCandidate.score >= maxScore || maxScore < scoreThreshold) {
                    continue;
                }

                newCandidate = {cv::Point2d(maxPos) + cv::Point2d(rect.tl()) +
                                    cv::Point2d(layerTemplate.rect.center),
                                layerTemplate.rect.angle, maxScore};
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

std::vector<Pose> matchModel(const cv::Mat &dst, const Model *model, int level, double startAngle,
                             double spanAngle, const double maxOverlap, const double minScore,
                             const int maxCount, const int subpixel) {
    (void)(subpixel);

    // prepare
    {
        if (dst.empty() || nullptr == model || model->layers.empty()) {
            return {};
        }

        if (dst.cols < model->srcSize.width || dst.rows < model->srcSize.height ||
            dst.size().area() < model->srcSize.area()) {
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
    const auto candidates =
        matchTopLevel(pyramids.back(), model->layers[ level ], model->startAngle, startAngle,
                      spanAngle, maxOverlap, minScore, maxCount, level);

    // match candidate each level
    auto levelMatched =
        matchDownLevel(pyramids, model->startAngle, candidates, minScore, subpixel, model, level);
    if (levelMatched.empty()) {
        return {};
    }

    // filter overlap
    {
        std::vector<cv::RotatedRect> rects;
        rects.reserve(levelMatched.size());
        for (const auto &candidate : levelMatched) {
            rects.emplace_back(cv::Point2f(candidate.pos), cv::Size2f(model->srcSize),
                               static_cast<float>(candidate.angle));
        }
        filterOverlap(levelMatched, rects, maxOverlap);
    }

    std::vector<Pose> result;
    for (const auto &candidate : levelMatched) {
        if (candidate.score < 0) {
            continue;
        }

        result.emplace_back(
            Pose{static_cast<float>(candidate.pos.x), static_cast<float>(candidate.pos.y),
                 static_cast<float>(candidate.angle), static_cast<float>(candidate.score)});

        if (result.size() >= static_cast<std::size_t>(maxCount)) {
            break;
        }
    }

    return result;
}
