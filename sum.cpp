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
    auto          *srcPtr    = src.data;
    cv::v_uint32x4 vSum      = cv::v_setzero_u32();
    cv::v_uint64x2 vSqSum    = cv::v_setzero_u64();
    uint32_t       partSum   = 0;
    uint64         partSqSum = 0;

    for (const auto &rle : hRegion) {
        auto *ptr = srcPtr + src.step * rle.row + rle.startColumn;

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

inline void computeSumDiff(const cv::v_uint16x8 &start, const cv::v_uint16x8 &end,
                           cv::v_int32x4 &diff0, cv::v_int32x4 &diff1) {
    cv::v_int16x8 sub;
    {
        auto vStart = cv::v_reinterpret_as_s16(start);
        auto vEnd   = cv::v_reinterpret_as_s16(end);
        sub         = cv::v_sub(vEnd, vStart);
    }

    cv::v_int32x4 val = cv::v_expand_low(sub);
    diff0             = cv::v_add(diff0, val);

    val   = cv::v_expand_high(sub);
    diff1 = cv::v_add(diff1, val);
}

inline void computeSumDiff(const cv::v_uint8x16 &start, const cv::v_uint8x16 &end,
                           cv::v_int32x4 &diff0, cv::v_int32x4 &diff1, cv::v_int32x4 &diff2,
                           cv::v_int32x4 &diff3) {
    computeSumDiff(cv::v_expand_low(start), cv::v_expand_low(end), diff0, diff1);
    computeSumDiff(cv::v_expand_high(start), cv::v_expand_high(end), diff2, diff3);
}

inline void computeSqSumDiff(const cv::v_uint32x4 &start, const cv::v_uint32x4 &end,
                             cv::v_int32x4 &diff0) {
    cv::v_int32x4 vStart = cv::v_reinterpret_as_s32(start);
    cv::v_int32x4 vEnd   = cv::v_reinterpret_as_s32(end);

    cv::v_int32x4 sub = cv::v_sub(vEnd, vStart);
    diff0             = cv::v_add(diff0, sub);
}

inline void computeSqSumDiff(cv::v_uint16x8 &start, cv::v_uint16x8 &end, cv::v_int32x4 &diff0,
                             cv::v_int32x4 &diff1) {
    start = cv::v_mul(start, start);
    end   = cv::v_mul(end, end);

    computeSqSumDiff(cv::v_expand_low(start), cv::v_expand_low(end), diff0);
    computeSqSumDiff(cv::v_expand_high(start), cv::v_expand_high(end), diff1);
}

inline void computeSqSumDiff(const cv::v_uint8x16 &start, const cv::v_uint8x16 &end,
                             cv::v_int32x4 &diff0, cv::v_int32x4 &diff1, cv::v_int32x4 &diff2,
                             cv::v_int32x4 &diff3) {

    auto vStart = cv::v_expand_low(start);
    auto vEnd   = cv::v_expand_low(end);
    computeSqSumDiff(vStart, vEnd, diff0, diff1);

    vStart = cv::v_expand_high(start);
    vEnd   = cv::v_expand_high(end);
    computeSqSumDiff(vStart, vEnd, diff2, diff3);
}

inline void v_expand_store(double *ptr, const std::array<int, 4> &val) {
    ptr[ 0 ] = ptr[ -1 ] + val[ 0 ];
    ptr[ 1 ] = ptr[ 0 ] + val[ 1 ];
    ptr[ 2 ] = ptr[ 1 ] + val[ 2 ];
    ptr[ 3 ] = ptr[ 2 ] + val[ 3 ];
}

void shiftH(const uchar *src, std::size_t srcStep, const HRegion &hRegion, int row, double *sum,
            std::size_t sumStep, int sumWidth, double *sqSum, std::size_t sqSumStep) {
    constexpr auto blockSize = simdSize(cv::v_uint8);
    auto          *srcPtr    = src;
    auto          *sumPtr    = sum + row * sumStep;
    auto          *sqSumPtr  = sqSum + row * sqSumStep;

    std::array<int, 4> buf;

    int i = 1;
    for (; i < sumWidth - blockSize; i += blockSize) {
        cv::v_int32x4 diff0 = cv::v_setzero_s32();
        cv::v_int32x4 diff1 = cv::v_setzero_s32();
        cv::v_int32x4 diff2 = cv::v_setzero_s32();
        cv::v_int32x4 diff3 = cv::v_setzero_s32();

        cv::v_int32x4 diff10 = cv::v_setzero_s32();
        cv::v_int32x4 diff11 = cv::v_setzero_s32();
        cv::v_int32x4 diff12 = cv::v_setzero_s32();
        cv::v_int32x4 diff13 = cv::v_setzero_s32();

        for (const auto &rle : hRegion) {
            auto *startPtr = srcPtr + (row + rle.row) * srcStep + rle.startColumn + i - 1;
            auto *endPtr   = startPtr + rle.length;

            auto vStart = cv::v_load(startPtr);
            auto vEnd   = cv::v_load(endPtr);
            computeSumDiff(vStart, vEnd, diff0, diff1, diff2, diff3);
            computeSqSumDiff(vStart, vEnd, diff10, diff11, diff12, diff13);
        }

        auto *sumPtrStart = sumPtr + i;
        cv::v_store(buf.data(), diff0);
        v_expand_store(sumPtrStart, buf);
        cv::v_store(buf.data(), diff1);
        v_expand_store(sumPtrStart + 4, buf);
        cv::v_store(buf.data(), diff2);
        v_expand_store(sumPtrStart + 8, buf);
        cv::v_store(buf.data(), diff3);
        v_expand_store(sumPtrStart + 12, buf);

        auto *sqSumPtrStart = sqSumPtr + i;
        cv::v_store(buf.data(), diff10);
        v_expand_store(sqSumPtrStart, buf);
        cv::v_store(buf.data(), diff11);
        v_expand_store(sqSumPtrStart + 4, buf);
        cv::v_store(buf.data(), diff12);
        v_expand_store(sqSumPtrStart + 8, buf);
        cv::v_store(buf.data(), diff13);
        v_expand_store(sqSumPtrStart + 12, buf);
    }

    for (; i < sumWidth; i++) {
        int32_t partSum   = 0;
        int32_t partSqSum = 0;
        for (const auto &rle : hRegion) {
            auto *startPtr = srcPtr + (row + rle.row) * srcStep + rle.startColumn + i - 1;
            auto *endPtr   = startPtr + rle.length;

            int32_t start  = *startPtr;
            int32_t end    = *endPtr;
            partSum       += end - start;
            partSqSum     += end * end - start * start;
        }

        auto *sumPtrStart   = sumPtr + i;
        sumPtrStart[ 0 ]    = sumPtrStart[ -1 ] + partSum;
        auto *sqSumPtrStart = sqSumPtr + i;
        sqSumPtrStart[ 0 ]  = sqSumPtrStart[ -1 ] + partSqSum;
    }
}

void shiftV(const uchar *src, std::size_t srcStep, const VRegion &vRegion, int row, double *sum,
            std::size_t sumStep, double *sqSum, std::size_t sqSumStep) {
    auto *srcPtr   = src;
    auto *sumPtr   = sum + row * sumStep;
    auto *sqSumPtr = sqSum + row * sqSumStep;

    int32_t partSum   = 0;
    int32_t partSqSum = 0;
    for (const auto &rle : vRegion) {
        auto *startPtr = srcPtr + (row + rle.startRow - 1) * srcStep + rle.col;
        auto *endPtr   = startPtr + rle.length * srcStep;

        int32_t start = *startPtr;
        int32_t end   = *endPtr;

        partSum   += end - start;
        partSqSum += end * end - start * start;
    }

    sumPtr[ 0 ]   = *(sumPtr - sumStep) + partSum;
    sqSumPtr[ 0 ] = *(sqSumPtr - sqSumStep) + partSqSum;
}

void integralSum(const cv::Mat &src, cv::Mat &sum, cv::Mat &sqSum, const cv::Size &templateSize,
                 const HRegion &hRegion, const VRegion &vRegion) {
    const auto size = src.size() - templateSize + cv::Size(1, 1);
    sum.create(size, CV_64FC1);
    sqSum.create(size, CV_64FC1);

    auto *srcPtr   = src.data;
    auto *sumPtr   = (double *)sum.data;
    auto *sqSumPtr = (double *)sqSum.data;

    // compute first
    uint64 sum0;
    uint64 sqSum0;
    computeSum(src, hRegion, sum0, sqSum0);
    sumPtr[ 0 ]   = static_cast<double>(sum0);
    sqSumPtr[ 0 ] = static_cast<double>(sqSum0);

    for (int y = 0; y < size.height; y++) {
        shiftH(srcPtr, src.step, hRegion, y, sumPtr, sum.step1(), sum.cols, sqSumPtr,
               sqSum.step1());
        if (y + 1 < size.height) {
            shiftV(srcPtr, src.step, vRegion, y + 1, sumPtr, sum.step1(), sqSumPtr, sqSum.step1());
        }
    }
}