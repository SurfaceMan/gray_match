#include "sum.h"

#include <opencv2/core/hal/intrin.hpp>

inline cv::v_uint32x4 v_add_expand(const cv::v_uint16x8 &src) {
    cv::v_uint32x4 low;
    cv::v_uint32x4 high;
    cv::v_expand(src, low, high);

    return cv::v_add(low, high);
}

inline cv::v_uint64x2 v_add_expand(const cv::v_uint32x4 &src) {
    cv::v_uint64x2 low;
    cv::v_uint64x2 high;
    cv::v_expand(src, low, high);

    return cv::v_add(low, high);
}

inline void computeSum(const cv::v_uint8x16 &src, cv::v_uint32x4 &sum, cv::v_uint64x2 &sqSum) {
    cv::v_uint16x8 low;
    cv::v_uint16x8 high;
    cv::v_expand(src, low, high);

    sum      = cv::v_add(sum, v_add_expand(cv::v_add(low, high)));
    auto dot = cv::v_dotprod_expand_fast(src, src);
    sqSum    = cv::v_add(sqSum, v_add_expand(dot));
}

void computeSum(const cv::Mat &src, const HRegion &hRegion, uint64 &sum, uint64 &sqSum) {
    constexpr auto blockSize = simdSize(cv::v_uint8);

    cv::v_uint32x4 vSum      = cv::v_setzero_u32();
    cv::v_uint64x2 vSqSum    = cv::v_setzero_u64();
    uint32_t       partSum   = 0;
    uint64         partSqSum = 0;

    for (const auto &rle : hRegion) {
        auto *ptr = src.ptr<uchar>(rle.row) + rle.startColumn;

        int i = 0;
        for (; i < rle.length - blockSize; i += blockSize) {
            computeSum(cv::v_load(ptr + i), vSum, vSqSum);
        }

        // TODO aligned fill 0
        for (; i < rle.length; i++) {
            auto val   = ptr[ i ];
            partSum   += val;
            partSqSum += (ushort)val * (ushort)val;
        }
    }

    sum   = cv::v_reduce_sum(vSum) + partSum;
    sqSum = cv::v_reduce_sum(vSqSum) + partSqSum;
}

inline void computeSumDiff(const cv::v_uint8x16 &start, const cv::v_uint8x16 &end,
                           cv::v_int32x4 &diff0, cv::v_int32x4 &diff1, cv::v_int32x4 &diff2,
                           cv::v_int32x4 &diff3) {}

inline void computeSqSumDiff(const cv::v_uint8x16 &start, const cv::v_uint8x16 &end,
                             cv::v_int32x4 &diff0, cv::v_int32x4 &diff1, cv::v_int32x4 &diff2,
                             cv::v_int32x4 &diff3) {}

inline void v_expand_store(double *ptr, const cv::v_int32x4 &val, double &base) {
    auto vBase = cv::v_setall_f64(base);
}

void shiftH(const cv::Mat &src, const HRegion &hRegion, int row, cv::Mat &sum, cv::Mat &sqSum) {
    constexpr auto blockSize = simdSize(cv::v_uint8);
    auto          *srcPtr    = src.ptr<uchar>();
    auto          *sumPtr    = sum.ptr<double>(row);
    auto          *sqSumPtr  = sqSum.ptr<double>(row);

    double sumBase   = *sumPtr;
    double sqSumBase = *sqSumPtr;

    int i = 1;
    for (; i < sum.cols - blockSize; i += blockSize) {
        cv::v_int32x4 diff0 = cv::v_setzero_s32();
        cv::v_int32x4 diff1 = cv::v_setzero_s32();
        cv::v_int32x4 diff2 = cv::v_setzero_s32();
        cv::v_int32x4 diff3 = cv::v_setzero_s32();

        cv::v_int32x4 diff10 = cv::v_setzero_s32();
        cv::v_int32x4 diff11 = cv::v_setzero_s32();
        cv::v_int32x4 diff12 = cv::v_setzero_s32();
        cv::v_int32x4 diff13 = cv::v_setzero_s32();

        for (const auto &rle : hRegion) {
            auto *startPtr = srcPtr + (row + rle.row) * src.step + rle.startColumn;
            auto *endPtr   = startPtr + rle.length;

            auto vStart = cv::v_load(startPtr);
            auto vEnd   = cv::v_load(endPtr);
            computeSumDiff(vStart, vEnd, diff0, diff1, diff2, diff3);
            computeSumDiff(vStart, vEnd, diff10, diff11, diff12, diff13);
        }

        auto *sumPtrStart = sumPtr + i;
        v_expand_store(sumPtrStart, diff0, sumBase);
        v_expand_store(sumPtrStart + 4, diff1, sumBase);
        v_expand_store(sumPtrStart + 8, diff2, sumBase);
        v_expand_store(sumPtrStart + 12, diff3, sumBase);

        auto *sqSumPtrStart = sqSumPtr + i;
        v_expand_store(sqSumPtrStart, diff10, sumBase);
        v_expand_store(sqSumPtrStart + 4, diff11, sumBase);
        v_expand_store(sqSumPtrStart + 8, diff12, sumBase);
        v_expand_store(sqSumPtrStart + 12, diff13, sumBase);
    }

    for (; i < sum.cols; i++) {}
}

void shiftV(const cv::Mat &src, const VRegion &vRegion, int row, cv::Mat &sum, cv::Mat &sqSum) {}

void integralSum(const cv::Mat &src, cv::Mat &sum, cv::Mat &sqSum, const cv::Size &templateSize,
                 const HRegion &hRegion, const VRegion &vRegion) {
    const auto size = src.size() - templateSize + cv::Size(1, 1);
    sum.create(size, CV_64FC1);
    sqSum.create(size, CV_64FC1);

    auto sumPtr   = sum.ptr<double>();
    auto sqSumPtr = sqSum.ptr<double>();

    // compute first
    uint64 sum0;
    uint64 sqSum0;
    computeSum(src, hRegion, sum0, sqSum0);
    sumPtr[ 0 ]   = static_cast<double>(sum0);
    sqSumPtr[ 0 ] = static_cast<double>(sqSum0);

    for (int y = 0; y < size.height; y++) {
        shiftH(src, hRegion, y, sum, sqSum);
        if (y + 1 < size.height) {
            shiftV(src, vRegion, y + 1, sum, sqSum);
        }
    }
}