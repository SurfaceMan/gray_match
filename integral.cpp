#include "integral.h"
#include "privateType.h"

#include <opencv2/core/hal/intrin.hpp>

inline void expand(const cv::v_int32 &src, cv::v_float64 &low, cv::v_float64 &high) {
    low  = cv::v_cvt_f64(src);
    high = cv::v_cvt_f64_high(src);
}

inline void integralSum(const cv::v_uint16 &src, double *dst, const double *prevDst,
                        cv::v_uint32 &pre) {
    auto sum = cv::v_add(src, cv::v_rotate_left<1>(src));
    sum      = cv::v_add(sum, cv::v_rotate_left<2>(sum));
    sum      = cv::v_add(sum, cv::v_rotate_left<4>(sum));

    cv::v_uint32 v1;
    cv::v_uint32 v2;
    cv::v_expand(sum, v1, v2);
    v1  = cv::v_add(v1, pre);
    v2  = cv::v_add(v2, pre);
    pre = cv::v_setall_u32(cv::v_extract_n<simdSize(cv::v_uint32) - 1>(v2));

    cv::v_float64 v3;
    cv::v_float64 v4;
    expand(cv::v_reinterpret_as_s32(v1), v3, v4);
    cv::v_store(dst, cv::v_add(v3, cv::v_load(prevDst)));
    cv::v_store(dst + simdSize(cv::v_float64),
                cv::v_add(v4, cv::v_load(prevDst + simdSize(cv::v_float64))));

    expand(cv::v_reinterpret_as_s32(v2), v3, v4);
    cv::v_store(dst + simdSize(cv::v_float64) * 2,
                cv::v_add(v3, cv::v_load(prevDst + simdSize(cv::v_float64) * 2)));
    cv::v_store(dst + simdSize(cv::v_float64) * 3,
                cv::v_add(v4, cv::v_load(prevDst + simdSize(cv::v_float64) * 3)));
}

inline void integralSqSum(cv::v_uint16 &src, double *dst, double *prevDst, cv::v_uint32 &pre) {
    cv::v_uint32 v1;
    cv::v_uint32 v2;
    cv::v_expand(src, v1, v2);

    {
        auto         shift1 = cv::v_rotate_left<1>(src);
        cv::v_uint32 v3;
        cv::v_uint32 v4;
        cv::v_expand(shift1, v3, v4);

        v1 = cv::v_add(v1, v3);
        v2 = cv::v_add(v2, v4);

        v4 = cv::v_extract<2>(v1, v2);
        v2 = cv::v_add(v2, v4);

        v3 = cv::v_rotate_left<2>(v1);
        v1 = cv::v_add(v1, v3);

        v1 = cv::v_add(v1, pre);
        v2 = cv::v_add(v2, v1);

        pre = cv::v_setall_u32(cv::v_extract_n<simdSize(cv::v_uint32) - 1>(v2));
    }

    cv::v_float64 v3;
    cv::v_float64 v4;
    expand(cv::v_reinterpret_as_s32(v1), v3, v4);
    cv::v_store(dst, cv::v_add(v3, cv::v_load(prevDst)));
    cv::v_store(dst + simdSize(cv::v_float64),
                cv::v_add(v4, cv::v_load(prevDst + simdSize(cv::v_float64))));

    expand(cv::v_reinterpret_as_s32(v2), v3, v4);
    cv::v_store(dst + simdSize(cv::v_float64) * 2,
                cv::v_add(v3, cv::v_load(prevDst + simdSize(cv::v_float64) * 2)));
    cv::v_store(dst + simdSize(cv::v_float64) * 3,
                cv::v_add(v4, cv::v_load(prevDst + simdSize(cv::v_float64) * 3)));
}

/*
inline void integralSqSum(cv::v_uint32 &src, double *dst, double *prevDst, cv::v_uint32 &pre) {
    src += cv::v_rotate_left<1>(src);
    src += cv::v_rotate_left<2>(src);
    src += pre;
    pre  = cv::v_setall_u32(cv::v_extract_n<simdSize(cv::v_uint32) - 1>(src));

    cv::v_float64 v1;
    cv::v_float64 v2;
    expand(cv::v_reinterpret_as_s32(src), v1, v2);

    cv::v_store(dst, v1 + cv::v_load(prevDst));
    cv::v_store(dst + simdSize(cv::v_float64), v2 + cv::v_load(prevDst +
simdSize(cv::v_float64)));
}

inline void integralSqSum(cv::v_uint16 &src, double *dst, double *prevDst, cv::v_uint32 &pre) {
    cv::v_uint32 v1;
    cv::v_uint32 v2;
    cv::v_expand(src, v1, v2);
    integralSqSum(v1, dst, prevDst, pre);
    integralSqSum(v2, dst + simdSize(cv::v_uint32), prevDst + simdSize(cv::v_uint32),
pre);
}
*/

inline void integralSum(const cv::v_uint16 &v1, const cv::v_uint16 &v2, double *dst,
                        const double *prevDst, cv::v_uint32 &pre) {
    integralSum(v1, dst, prevDst, pre);
    integralSum(v2, dst + simdSize(cv::v_uint16), prevDst + simdSize(cv::v_uint16), pre);
}

inline void integralSqSum(cv::v_uint16 &v1, cv::v_uint16 &v2, double *dst, double *prevDst,
                          cv::v_uint32 &pre) {
    v1 = cv::v_mul_wrap(v1, v1);
    v2 = cv::v_mul_wrap(v2, v2);

    integralSqSum(v1, dst, prevDst, pre);
    integralSqSum(v2, dst + simdSize(cv::v_uint16), prevDst + simdSize(cv::v_uint16), pre);
}

void integralSimd(const cv::Mat &src, cv::Mat &sum, cv::Mat &sqSum) {
    const auto size = src.size() + cv::Size(1, 1);
    sum.create(size, CV_64FC1);
    sqSum.create(size, CV_64FC1);
    memset(sum.data, 0, sum.step[ 0 ]);
    memset(sqSum.data, 0, sqSum.step[ 0 ]);

    const auto *srcStart   = src.data;
    const auto  srcStep    = src.step[ 0 ];
    auto       *sumStart   = reinterpret_cast<double *>(sum.data) + sum.step1() + 1;
    const auto  sumStep    = sum.step[ 0 ] / sum.step[ 1 ];
    auto       *sqSumStart = reinterpret_cast<double *>(sqSum.data) + sqSum.step1() + 1;
    const auto  sqSumStep  = sqSum.step[ 0 ] / sqSum.step[ 1 ];
    const auto  end        = size.width - simdSize(cv::v_uint8);
    for (int y = 0; y < src.rows; y++) {
        auto       *srcPtr    = srcStart + srcStep * y;
        auto       *sumPtr    = sumStart + sumStep * y;
        const auto *preSumPtr = sumStart + sumStep * (y - 1);
        sumPtr[ -1 ]          = 0;

        cv::v_uint32 prevSum = cv::vx_setzero_u32();
        for (int x = 0; x < end; x += simdSize(cv::v_uint8)) {
            cv::v_uint16 v1;
            cv::v_uint16 v2;
            cv::v_expand(cv::v_load(srcPtr + x), v1, v2);

            integralSum(v1, v2, sumPtr + x, preSumPtr + x, prevSum);
        }
    }

    for (int y = 0; y < src.rows; y++) {
        auto *srcPtr      = srcStart + srcStep * y;
        auto *sqSumPtr    = sqSumStart + sqSumStep * y;
        auto *preSqSumPtr = sqSumStart + sqSumStep * (y - 1);
        sqSumPtr[ -1 ]    = 0;

        cv::v_uint32 prevSqSum = cv::vx_setzero_u32();
        for (int x = 0; x < end; x += simdSize(cv::v_uint8)) {
            cv::v_uint16 v1;
            cv::v_uint16 v2;
            cv::v_expand(cv::v_load(srcPtr + x), v1, v2);

            integralSqSum(v1, v2, sqSumPtr + x, preSqSumPtr + x, prevSqSum);
        }
    }

    const auto start = src.cols - src.cols % simdSize(cv::v_uint8);
    for (int y = 0; y < src.rows; y++) {
        auto       *srcPtr      = srcStart + srcStep * y;
        auto       *sumPtr      = sumStart + sumStep * y;
        auto       *sqSumPtr    = sqSumStart + sqSumStep * y;
        const auto *preSumPtr   = sumStart + sumStep * (y - 1);
        const auto *preSqSumPtr = sqSumStart + sqSumStep * (y - 1);
        for (int x = start; x < src.cols; x++) {
            const auto val   = srcPtr[ x ];
            const auto sqVal = val * val;

            sumPtr[ x ]   = sumPtr[ x - 1 ] + val + preSumPtr[ x ] - preSumPtr[ x - 1 ];
            sqSumPtr[ x ] = sqSumPtr[ x - 1 ] + sqVal + preSqSumPtr[ x ] - preSqSumPtr[ x - 1 ];
        }
    }
}
