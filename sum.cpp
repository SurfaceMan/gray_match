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

inline void computeSumDiff(const cv::v_uint16x8 &start, const cv::v_uint16x8 &end,
                           cv::v_int32x4 &diff0, cv::v_int32x4 &diff1, cv::v_int32x4 &base) {
    cv::v_int16x8 sub;
    {
        cv::v_int16x8 vStart = cv::v_reinterpret_as_s16(start);
        cv::v_int16x8 vEnd   = cv::v_reinterpret_as_s16(end);
        sub                  = cv::v_sub(vEnd, vStart);
        sub                  = cv::v_add(sub, cv::v_rotate_left<1>(sub));
        sub                  = cv::v_add(sub, cv::v_rotate_left<2>(sub));
        sub                  = cv::v_add(sub, cv::v_rotate_left<4>(sub));
    }

    cv::v_int32x4 val = cv::v_expand_low(sub);
    val               = cv::v_add(val, base);
    diff0             = cv::v_add(diff0, val);
    base              = cv::v_setall_s32(cv::v_extract_n<simdSize(cv::v_int32) - 1>(diff0));

    val   = cv::v_expand_high(sub);
    val   = cv::v_add(val, base);
    diff1 = cv::v_add(diff1, val);
    base  = cv::v_setall_s32(cv::v_extract_highest(diff1));
}

inline void computeSumDiff(const cv::v_uint8x16 &start, const cv::v_uint8x16 &end,
                           cv::v_int32x4 &diff0, cv::v_int32x4 &diff1, cv::v_int32x4 &diff2,
                           cv::v_int32x4 &diff3) {
    auto base = cv::v_setzero_s32();
    computeSumDiff(cv::v_expand_low(start), cv::v_expand_low(end), diff0, diff1, base);
    computeSumDiff(cv::v_expand_high(start), cv::v_expand_high(end), diff2, diff3, base);
}

inline void computeSqSumDiff(const cv::v_uint32x4 &start, const cv::v_uint32x4 &end,
                             cv::v_int32x4 &diff0, cv::v_int32x4 &base) {
    cv::v_int32x4 vStart = cv::v_reinterpret_as_s32(start);
    cv::v_int32x4 vEnd   = cv::v_reinterpret_as_s32(end);

    cv::v_int32x4 sub = cv::v_sub(vEnd, vStart);
    sub               = cv::v_add(sub, cv::v_rotate_left<1>(sub));
    sub               = cv::v_add(sub, cv::v_rotate_left<2>(sub));

    sub   = cv::v_add(sub, base);
    diff0 = cv::v_add(diff0, sub);
    base  = cv::v_setall_s32(cv::v_extract_highest(diff0));
}

inline void computeSqSumDiff(cv::v_uint16x8 &start, cv::v_uint16x8 &end, cv::v_int32x4 &diff0,
                             cv::v_int32x4 &diff1, cv::v_int32x4 &base) {
    start = cv::v_mul(start, start);
    end   = cv::v_mul(end, end);

    computeSqSumDiff(cv::v_expand_low(start), cv::v_expand_low(end), diff0, base);
    computeSqSumDiff(cv::v_expand_high(start), cv::v_expand_high(end), diff1, base);
}

inline void computeSqSumDiff(const cv::v_uint8x16 &start, const cv::v_uint8x16 &end,
                             cv::v_int32x4 &diff0, cv::v_int32x4 &diff1, cv::v_int32x4 &diff2,
                             cv::v_int32x4 &diff3) {
    auto base = cv::v_setzero_s32();

    auto vStart = cv::v_expand_low(start);
    auto vEnd   = cv::v_expand_low(end);
    computeSqSumDiff(vStart, vEnd, diff0, diff1, base);

    vStart = cv::v_expand_high(start);
    vEnd   = cv::v_expand_high(end);
    computeSqSumDiff(vStart, vEnd, diff2, diff3, base);
}

inline void v_expand_store(double *ptr, const cv::v_int32x4 &val, double &base) {
    auto vBase = cv::v_setall_f64(base);

    auto tmp = cv::v_reinterpret_as_f64(cv::v_expand_low(val));
    tmp      = cv::v_add(tmp, vBase);
    cv::v_store(ptr, tmp);
    vBase = cv::v_setall_f64(cv::v_extract_highest(tmp));

    tmp = cv::v_reinterpret_as_f64(cv::v_expand_high(val));
    tmp = cv::v_add(tmp, vBase);
    cv::v_store(ptr + 2, tmp);

    base = cv::v_extract_highest(tmp);
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
            auto *startPtr = srcPtr + (row + rle.row) * src.step + rle.startColumn + i;
            auto *endPtr   = startPtr + rle.length;

            auto vStart = cv::v_load(startPtr);
            auto vEnd   = cv::v_load(endPtr);
            computeSumDiff(vStart, vEnd, diff0, diff1, diff2, diff3);
            computeSqSumDiff(vStart, vEnd, diff10, diff11, diff12, diff13);
        }

        auto *sumPtrStart = sumPtr + i;
        v_expand_store(sumPtrStart, diff0, sumBase);
        v_expand_store(sumPtrStart + 4, diff1, sumBase);
        v_expand_store(sumPtrStart + 8, diff2, sumBase);
        v_expand_store(sumPtrStart + 12, diff3, sumBase);

        auto *sqSumPtrStart = sqSumPtr + i;
        v_expand_store(sqSumPtrStart, diff10, sqSumBase);
        v_expand_store(sqSumPtrStart + 4, diff11, sqSumBase);
        v_expand_store(sqSumPtrStart + 8, diff12, sqSumBase);
        v_expand_store(sqSumPtrStart + 12, diff13, sqSumBase);
    }

    for (; i < sum.cols; i++) {
        int32_t partSum   = 0;
        int32_t partSqSum = 0;
        for (const auto &rle : hRegion) {
            auto *startPtr = srcPtr + (row + rle.row) * src.step + rle.startColumn + i;
            auto *endPtr   = startPtr + rle.length;

            auto start  = *startPtr;
            auto end    = *endPtr;
            partSum    += end - start;
            partSqSum  += end * end - start * start;
        }

        auto *sumPtrStart   = sumPtr + i;
        sumPtrStart[ 0 ]    = sumPtrStart[ -1 ] + partSum;
        auto *sqSumPtrStart = sqSumPtr + i;
        sqSumPtrStart[ 0 ]  = sqSumPtrStart[ -1 ] + partSqSum;
    }
}

void shiftV(const cv::Mat &src, const VRegion &vRegion, int row, cv::Mat &sum, cv::Mat &sqSum) {
    auto *srcPtr   = src.ptr<uchar>();
    auto *sumPtr   = sum.ptr<double>(row);
    auto *sqSumPtr = sqSum.ptr<double>(row);

    int32_t partSum   = 0;
    int32_t partSqSum = 0;
    for (const auto &rle : vRegion) {
        auto *startPtr = srcPtr + (row + rle.startRow) * src.step + rle.col;
        auto *endPtr   = startPtr + rle.length * src.step;

        int32_t start = *startPtr;
        int32_t end   = *endPtr;

        partSum   += end - start;
        partSqSum += end * end - start * start;
    }

    sumPtr[ 0 ]   = *(sumPtr - sum.step1()) + partSum;
    sqSumPtr[ 0 ] = *(sqSumPtr - sum.step1()) + partSqSum;
}

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