#include "grayMatch.h"

#include <opencv2/core/hal/intrin.hpp>

const int    MIN_AREA  = 256;
const double TOLERANCE = 0.0000001;
const int    CANDIDATE = 5;
const double INVALID   = -1.;

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

    void resize(std::size_t size) {
        normal.resize(size);
        invArea.resize(size);
        mean.resize(size);
        equal1.resize(size);
    }

    void reserve(std::size_t size) {
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

        Block(const cv::Rect &rect_, const cv::Point &maxPos_, double maxScore_)
            : rect(rect_)
            , maxPos(maxPos_)
            , maxScore(maxScore_) {}

        bool operator<(const Block &rhs) const {
            return this->maxScore > rhs.maxScore;
        }
    };

    std::vector<Block> blocks;
    cv::Mat            score;

    BlockMax(cv::Mat score_, cv::Size templateSize) {
        score = score_;

        // divide source image to blocks then compute max
        auto blockWidth  = templateSize.width * 2;
        auto blockHeight = templateSize.height * 2;

        auto nWidth    = score.size().width / blockWidth;
        auto nHeight   = score.size().height / blockHeight;
        auto hRemained = score.size().width % blockWidth;
        auto vRemained = score.size().height % blockHeight;

        blocks.resize(nWidth * nHeight);
        int i = 0;
        for (int y = 0; y < nHeight; y++) {
            for (int x = 0; x < nWidth; x++) {
                cv::Rect rect(x * blockWidth, y * blockHeight, blockWidth, blockHeight);

                auto &block = blocks[ i ];
                block.rect  = rect;
                cv::minMaxLoc(score(rect), 0, &block.maxScore, 0, &block.maxPos);
                block.maxPos += rect.tl();
            }
        }

        if (hRemained) {
            cv::Rect rightRect(nWidth * blockWidth, 0, hRemained, score.size().height);
            Block    rightBlock;
            rightBlock.rect = rightRect;
            cv::minMaxLoc(score(rightRect), 0, &rightBlock.maxScore, 0, &rightBlock.maxPos);
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
            cv::minMaxLoc(score(bottomRect), 0, &bottomBlock.maxScore, 0, &bottomBlock.maxPos);
            bottomBlock.maxPos += bottomRect.tl();
            blocks.push_back(std::move(bottomBlock));
        }
    }

    void update(cv::Rect rect) {
        for (auto &block : blocks) {
            auto intersection = block.rect & rect;
            if (intersection.empty()) {
                continue;
            }

            // update
            cv::minMaxLoc(score(block.rect), 0, &block.maxScore, 0, 0);
        }
    }

    void maxValueLoc(double &maxScore, cv::Point &maxPos) {
        auto max = std::max_element(blocks.begin(), blocks.end());
        maxScore = max->maxScore;
        maxPos   = max->maxPos;
    }
};

struct Candidate {
    cv::Point2d pos;
    double      angle;
    double      score;

    Candidate()
        : angle(0)
        , score(0) {}

    Candidate(const cv::Point2d &pos_, double angle_, double score_)
        : pos(pos_)
        , angle(angle_)
        , score(score_) {}

    bool operator<(const Candidate &rhs) const {
        return this->score > rhs.score;
    }
};

int computeLayers(int width, int height, int minArea) {
    assert(width > 0 && height > 0 && minArea > 0);

    auto area  = width * height;
    int  layer = 0;
    while (area > minArea) {
        area /= 4;
        layer++;
    }

    return layer;
}

cv::Size computeRotationSize(const cv::Size &dstSize, const cv::Size &templateSize, double angle,
                             const cv::Mat &rotate) {
    if (angle > 360) {
        angle -= 360;
    } else if (angle < 0) {
        angle += 360;
    }

    if (fabs(fabs(angle) - 90) < TOLERANCE || fabs(fabs(angle) - 270) < TOLERANCE) {
        return cv::Size(dstSize.height, dstSize.width);
    } else if (fabs(angle) < TOLERANCE || fabs(fabs(angle) - 180) < TOLERANCE) {
        return dstSize;
    }

    std::vector<cv::Point2d> points{{0, 0},
                                    {(double)dstSize.width - 1, 0},
                                    {0, (double)dstSize.height - 1},
                                    {(double)dstSize.width - 1, (double)dstSize.height - 1}};
    std::vector<cv::Point2d> trans;
    cv::transform(points, trans, rotate);

    cv::Point2d min = trans[ 0 ];
    cv::Point2d max = trans[ 0 ];
    for (const auto &point : trans) {
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

    if (angle > 0 && angle < 90) {
        ;
    } else if (angle > 90 && angle < 180) {
        angle -= 90;
    } else if (angle > 180 && angle < 270) {
        angle -= 180;
    } else if (angle > 270 && angle < 360) {
        angle -= 270;
    }

    auto radius = angle / 180. * CV_PI;
    auto dy     = sin(radius);
    auto dx     = cos(radius);
    auto width  = templateSize.width * dx * dy;
    auto height = templateSize.height * dx * dy;

    auto center     = cv::Point2d((dstSize.width - 1.) / 2., (dstSize.height - 1.) / 2.);
    auto halfHeight = static_cast<int>(ceil(max.y - center.y - width));
    auto halfWidth  = static_cast<int>(ceil(max.x - center.x - height));

    cv::Size size(halfWidth * 2, halfHeight * 2);
    auto     wrongSize = (templateSize.width < size.width && templateSize.height > size.height) ||
                      (templateSize.width > size.width && templateSize.height < size.height) ||
                      templateSize.area() > size.area();
    if (wrongSize) {
        size = {int(max.x - min.x + 0.5), int(max.y - min.y + 0.5)};
    }

    return size;
}

void coeffDenominator(const cv::Mat &src, const cv::Size &templateSize, cv::Mat &result, double mean,
                      double normal, double invArea, bool equal1) {
    if (equal1) {
        result = cv::Scalar::all(1);
        return;
    }

    cv::Mat sum;
    cv::Mat sqSum;
    cv::integral(src, sum, sqSum, CV_64F);

    auto *q0 = sqSum.ptr<double>(0);
    auto *q1 = q0 + templateSize.width;
    auto *q2 = sqSum.ptr<double>(templateSize.height);
    auto *q3 = q2 + templateSize.width;

    auto *p0 = sum.ptr<double>(0);
    auto *p1 = p0 + templateSize.width;
    auto *p2 = sum.ptr<double>(templateSize.height);
    auto *p3 = p2 + templateSize.width;

    auto step = sum.step / sizeof(double);

    for (int y = 0; y < result.rows; y++) {
        auto *scorePtr = result.ptr<float>(y);
        auto  idx      = y * step;
        for (int x = 0; x < result.cols; x++, idx++) {
            auto &score     = scorePtr[ x ];
            auto  partSum   = p0[ idx ] - p1[ idx ] - p2[ idx ] + p3[ idx ];
            auto  partMean  = partSum * partSum;
            auto  num       = score - partSum * mean;
            partMean       *= invArea;

            auto partSum2 = q0[ idx ] - q1[ idx ] - q2[ idx ] + q3[ idx ];

            auto diff = std::max(partSum2 - partMean, 0.);
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

float convSimd(uchar* kernel, uchar* src, int kernelWidth){
    auto blockSize = cv::VTraits<cv::v_uint8>::vlanes();
    auto vSum = cv::vx_setall_u32(0) ;
    int i = 0;
    for(; i < kernelWidth - blockSize; i += blockSize){
      vSum += cv::v_dotprod_expand(cv::v_load(kernel+i), cv::v_load(src+i));
    }
    auto sum = cv::v_reduce_sum(vSum);

    for(;i<kernelWidth; i++){
        sum += kernel[i]*src[i];
    }

    return (float)sum;
}

void matchTemplateSimd(cv::Mat &src, cv::Mat &templateImg, cv::Mat &result) {
    result = cv::Mat::zeros(src.size() - templateImg.size() + cv::Size(1, 1), CV_32FC1);

    for(int y = 0; y < src.rows; y++){
        auto *resultPtr = result.ptr<float>(y);
        for(int x = 0; x < src.cols; x++){
            auto &score = resultPtr[x];
            for(int templateRow = 0; templateRow < templateImg.rows; templateRow++){
                auto* srcPtr = src.ptr<uchar>(y + templateRow) + x;
                auto* temPtr = templateImg.ptr<uchar>(templateRow);
                score += convSimd(temPtr, srcPtr, templateImg.cols);
            }
        }
    }
}

void matchTemplate(cv::Mat &src, cv::Mat &result, Model *model, int level) {
#ifdef CV_SIMD
    matchTemplateSimd(src, model->pyramids[ level ], result);
#else
    cv::matchTemplate(src, model->pyramids[ level ], result, cv::TM_CCORR);
#endif
    coeffDenominator(src, model->pyramids[ level ].size(), result, model->mean[ level ][ 0 ],
                     model->normal[ level ], model->invArea[ level ], model->equal1[ level ]);
}

void nextMaxLoc(const cv::Point &pos, cv::Size templateSize, double maxOverlap, BlockMax &block,
                double &maxScore, cv::Point &maxPos) {
    auto      alone = 1. - maxOverlap;
    cv::Point offset(int(templateSize.width * alone), int(templateSize.height * alone));
    cv::Size  size(int(2 * templateSize.width * alone), int(2 * templateSize.height * alone));
    cv::Rect  rectIgnore(pos - offset, size);

    // clear neighbor
    cv::rectangle(block.score, rectIgnore, cv::Scalar(-1), cv::FILLED);

    block.update(rectIgnore);
    block.maxValueLoc(maxScore, maxPos);
}

void nextMaxLoc(cv::Mat &score, const cv::Point &pos, cv::Size templateSize, double maxOverlap,
                double &maxScore, cv::Point &maxPos) {
    auto      alone = 1. - maxOverlap;
    cv::Point offset(int(templateSize.width * alone), int(templateSize.height * alone));
    cv::Size  size(int(2 * templateSize.width * alone), int(2 * templateSize.height * alone));
    cv::Rect  rectIgnore(pos - offset, size);

    // clear neighbor
    cv::rectangle(score, rectIgnore, cv::Scalar(-1), cv::FILLED);

    cv::minMaxLoc(score, 0, &maxScore, 0, &maxPos);
}

inline cv::Point2d transform(const cv::Point2d &point, const cv::Mat &rotate) {
    auto ptr = rotate.ptr<double>();

    auto x = point.x * ptr[ 0 ] + point.y * ptr[ 1 ] + ptr[ 2 ];
    auto y = point.x * ptr[ 3 ] + point.y * ptr[ 4 ] + ptr[ 5 ];

    return {x, y};
}

inline cv::Point2d transform(const cv::Point2d &point, const cv::Point &center, double angle) {
    auto rotate = cv::getRotationMatrix2D(center, angle, 1.);

    return transform(point, rotate);
}

inline cv::Point2d sizeCenter(const cv::Size &size) {
    return cv::Point2d((size.width - 1.) / 2., (size.height - 1.) / 2.);
}

void cropRotatedRoi(const cv::Mat &src, const cv::Size &templateSize, cv::Point2d topLeft,
                    const cv::Mat &rotate, cv::Mat &roi) {
    auto     point = transform(topLeft, rotate);
    cv::Size paddingSize(templateSize.width + 6, templateSize.height + 6);
    auto     rt          = rotate;
    rt.at<double>(0, 2) -= point.x - 3;
    rt.at<double>(1, 2) -= point.y - 3;

    cv::warpAffine(src, roi, rt, paddingSize);
}

void filterOverlap(std::vector<Candidate> &candidates, std::vector<cv::RotatedRect> &rects,
                   double maxOverlap) {
    auto size = candidates.size();
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
            auto                     type = cv::rotatedRectangleIntersection(rect, refRect, points);

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

                    auto area    = cv::contourArea(points);
                    auto overlap = area / rect.size.area();
                    if (overlap > maxOverlap) {
                        (candidate.score > refCandidate.score ? refCandidate.score
                                                              : candidate.score) = INVALID;
                    }
                }
            }
        }
    }
}

Model *trainModel(const cv::Mat &src, int level) {
    if (src.empty() || src.channels() != 1) {
        return nullptr;
    }

    if (level < 0) {
        // level must grater than 1
        level = computeLayers(src.size().width, src.size().height, MIN_AREA);
    }

    auto scale   = 1 << (level - 1);
    auto topArea = src.size().area() / (scale * scale);
    if (MIN_AREA > topArea) {
        // top area must greater than MIN_AREA
        return nullptr;
    }

    auto  *result = new Model;
    Model &model  = *result;
    cv::buildPyramid(src, model.pyramids, level);
    model.borderColor = cv::mean(src).val[ 0 ] < 128 ? 255 : 0;
    model.reserve(model.pyramids.size());

    for (const auto &pyramid : model.pyramids) {
        auto invArea = 1. / pyramid.size().area();

        cv::Scalar mean;
        cv::Scalar stdDev;
        cv::meanStdDev(pyramid, mean, stdDev);

        auto stdNormal = stdDev[ 0 ] * stdDev[ 0 ] + stdDev[ 1 ] * stdDev[ 1 ] +
                         stdDev[ 2 ] * stdDev[ 2 ] + stdDev[ 3 ] * stdDev[ 3 ];
        auto equal1 = stdNormal < std::numeric_limits<double>::epsilon();

        auto normal2 = stdNormal + mean[ 0 ] * mean[ 0 ] + mean[ 1 ] * mean[ 1 ] +
                       mean[ 2 ] * mean[ 2 ] + mean[ 3 ] * mean[ 3 ];
        normal2     /= invArea;
        auto normal  = sqrt(stdNormal) / sqrt(invArea);

        model.equal1.push_back(equal1);
        model.invArea.push_back(invArea);
        model.mean.push_back(mean);
        model.normal.push_back(normal);
    }

    return result;
}

std::vector<Pose> matchModel(const cv::Mat &dst, Model *model, int level, double startAngle,
                             double spanAngle, double maxOverlap, double minScore, int maxCount,
                             int subpixel) {
    //prepare
    {
        if (dst.empty() || nullptr == model) {
            return {};
        }

        auto &templateImg = model->pyramids.front();
        if (dst.cols < templateImg.cols || dst.rows < templateImg.rows ||
                dst.size().area() < templateImg.size().area()) {
            return {};
        }

        auto templateLevel = static_cast<int>(model->pyramids.size() - 1);
        if (level < 0 || level > templateLevel) {
            // level must grater than 1
            level = templateLevel;
        }
    }

    std::vector<cv::Mat> pyramids;
    cv::buildPyramid(dst, pyramids, level);

    // compute top
    std::vector<Candidate> candidates;
    {
        const auto &templateTop = model->pyramids[ level ];
        auto        angleStep = atan(2. / std::max(templateTop.cols, templateTop.rows)) * 180. / CV_PI;

        const auto &dstTop = pyramids.back();
        cv::Point2d center = sizeCenter(dstTop.size());
        bool calMaxByBlock = (dstTop.size().area() / templateTop.size().area() > 500) && maxCount > 10;
        const auto topScoreThreshold = minScore * pow(0.9, level);

        for (auto angle = startAngle; angle < startAngle + spanAngle + angleStep; angle += angleStep) {
            auto rotate = cv::getRotationMatrix2D(center, angle, 1.);
            auto size   = computeRotationSize(dstTop.size(), templateTop.size(), angle, rotate);

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
                cv::minMaxLoc(result, 0, &maxScore, 0, &maxPos);
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
    }

    // match candidate each level
    std::vector<Candidate> levelMatched;
    for (const auto &candidate : candidates) {
        auto pose    = candidate;
        bool matched = true;
        for (int currentLevel = level - 1; currentLevel >= 0; currentLevel--) {
            const auto &currentTemplateImg = model->pyramids[ currentLevel ];
            const auto  tmpSize            = currentTemplateImg.size();

            const auto &currentDstImg = pyramids[ currentLevel ];
            const auto  dstSize       = currentDstImg.size();

            auto currentAngleStep =
                atan(2. / std::max(tmpSize.width, tmpSize.height)) * 180. / CV_PI;
            auto center = sizeCenter(dstSize);

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
                cv::minMaxLoc(result, 0, &maxScore, 0, &maxPos);

                if (newCandidate.score >= maxScore || maxScore < scoreThreshold) {
                    continue;
                }

                newCandidate  = {cv::Point2d(maxPos), angle, maxScore};
                auto isBorder = 0 == maxPos.x || 0 == maxPos.y || result.cols - 1 == maxPos.x ||
                                result.rows - 1 == maxPos.y;
                newScoreRect = isBorder
                                   ? cv::Mat()
                                   : result(cv::Rect(maxPos - cv::Point{1, 1}, cv::Size(3, 3)));
            }

            if (newCandidate.score < scoreThreshold) {
                matched = false;
                break;
            }

            if (!newScoreRect.empty() && subpixel) {
                // TODO subpixel
            }

            // back to
            auto paddingTopLeft =
                transform(topLeft, center, newCandidate.angle) - cv::Point2d(3, 3);
            newCandidate.pos += paddingTopLeft;

            pose = std::move(newCandidate);
        }

        if (!matched) {
            continue;
        }

        const auto lastSize   = pyramids.front().size();
        auto       lastCenter = sizeCenter(lastSize);
        pose.pos              = transform(pose.pos, lastCenter, -pose.angle);

        levelMatched.push_back(std::move(pose));
    }
    std::sort(levelMatched.begin(), levelMatched.end());

    // filter overlap
    std::vector<cv::RotatedRect> rects;
    {
        rects.reserve(levelMatched.size());
        auto        size = model->pyramids.front().size();
        cv::Point2f topRight((float)size.width, 0.f);
        cv::Point2f bottomRight((float)size.width, (float)size.height);
        for (const auto &candidate : levelMatched) {
            std::vector<cv::Point2f> points{topRight + cv::Point2f(candidate.pos),
                        bottomRight + cv::Point2f(candidate.pos)};
            auto rotate = cv::getRotationMatrix2D(candidate.pos, -candidate.angle, 1.);
            std::vector<cv::Point2f> rotatedPoints;
            cv::transform(points, rotatedPoints, rotate);

            rects.emplace_back(
                        cv::RotatedRect{cv::Point2f(candidate.pos), rotatedPoints[ 0 ], rotatedPoints[ 1 ]});
        }
        filterOverlap(levelMatched, rects, maxOverlap);
    }

    std::vector<Pose> result;
    {
        auto count = levelMatched.size();
        for (std::size_t i = 0; i < count; i++) {
            auto &candidate = levelMatched[ i ];
            auto &rect      = rects[ i ];

            if (candidate.score < 0) {
                continue;
            }

            auto center = rect.center;
            result.emplace_back(
                        Pose{center.x, center.y, (float)-candidate.angle, (float)candidate.score});
        }

        std::sort(result.begin(), result.end(),
                  [](const Pose &a, const Pose &b) { return a.score > b.score; });
    }

    return result;
}
