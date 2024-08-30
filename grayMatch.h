#ifndef GRAY_MATCH_H
#define GRAY_MATCH_H

#include "apiExport.h"

struct Model;

using Model_t = Model *;

struct Pose {
    float x;
    float y;
    float angle;
    float score;
};

/**
 * @brief train match model
 * @param data image data
 * @param width image width
 * @param height image height
 * @param channels image channels 1(gray)/3(rgb)/4(rgba)
 * @param bytesPerline bytes per line
 * @param roiLeft rectangle roi left
 * @param roiTop rectangle roi top
 * @param roiWidth rectangle roi width
 * @param roiHeight rectangle roi height
 * @param levelNum pyramid levels (> 0)
 * @return
 */
API_PUBLIC Model_t trainModel(const unsigned char *data, int width, int height, int channels,
                              int bytesPerline, int roiLeft, int roiTop, int roiWidth,
                              int roiHeight, int levelNum);
/**
 * @brief match model
 * @param data image data
 * @param width image width
 * @param height image height
 * @param channels image channels 1(gray)/3(rgb)/4(rgba)
 * @param bytesPerline bytes per line
 * @param roiLeft rectangle roi left
 * @param roiTop rectangle roi top
 * @param roiWidth rectangle roi width
 * @param roiHeight rectangle roi height
 * @param model trained model
 * @param count in(max detect count)/out(found count)
 * @param poses pose array inited with size not less than count
 * @param level match start at which level (level>=0 && level<modelLevel-1)
 * @param startAngle rotation start angle
 * @param spanAngle rotation angle range
 * @param maxOverlap overlap threshold
 * @param minScore minimum matched score
 * @param subpixel compute subpixel result
 * @return
 */
API_PUBLIC void matchModel(const unsigned char *data, int width, int height, int channels,
                           int bytesPerline, int roiLeft, int roiTop, int roiWidth, int roiHeight,
                           const Model_t model, int *count, Pose *poses, int level,
                           double startAngle, double spanAngle, double maxOverlap, double minScore,
                           int subpixel);

/**
 * @brief get trained model levels
 * @param model
 * @return pyramid level
 */
API_PUBLIC int modelLevel(const Model_t model);

/**
 * @brief get trained model image
 * @param model
 * @param level pyramid level index(level>=0 && level<modelLevel-1)
 * @param data image data buffer(need allocated), can input nullptr to query width/height/channels
 * @param length buffer length not less than width*height*channels
 * @param width image width,  can input nullptr
 * @param height image height, can input nullptr
 * @param channels image channels, can input nullptr
 * @return
 */
API_PUBLIC void modelImage(const Model_t model, int level, unsigned char *data, int length,
                           int *width, int *height, int *channels);

/**
 * @brief free model
 * @param model
 * @return
 */
API_PUBLIC void freeModel(Model_t *model);

/**
 * @brief serialize model to buffer
 * @param model
 * @param buffer need allocated, can input nullptr to query size
 * @param size in(buffer size)/out(written size)
 * @return true(success)false(failed)
 */
API_PUBLIC bool serialize(const Model_t model, unsigned char *buffer, int *size);

/**
 * @brief desrialize model
 * @param buffer
 * @param size buffer size
 * @return model
 */
API_PUBLIC Model_t deserialize(unsigned char *buffer, int size);

#endif // GRAY_MATCH_H
