#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

int32_t _opus_application_audio();
int32_t _opus_application_voip();
int32_t _opus_application_lowdelay();

int32_t _opus_bitrate_max();
int32_t _opus_bitrate_auto();

void *_opus_encoder_create(int32_t fs, int32_t channels, int32_t application, int32_t *err);
void _opus_encoder_destroy(void *enc);
int32_t _opus_set_bitrate(void *enc, int32_t bitrate);
int32_t _opus_set_complexity(void *enc, int32_t complexity);
int32_t _opus_encode_i16(void *enc, const int16_t *pcm, size_t frame_size, uint8_t *data, size_t max_data_bytes);
int32_t _opus_encode_float(void *enc, const float *pcm, size_t frame_size, uint8_t *data, size_t max_data_bytes);

void *_opus_decoder_create(int32_t fs, int32_t channels, int32_t *error);
void _opus_decoder_destroy(void *dec);
int32_t _opus_decode_i16(void *dec, const uint8_t *data, size_t len, int16_t *pcm, size_t frame_size, int32_t decode_fec);
int32_t _opus_decode_float(void *dec, const uint8_t *data, size_t len, float *pcm, size_t frame_size, int32_t decode_fec);

const char *_opus_get_version_string(void);
const char *_opus_strerror(int32_t error);

#ifdef __cplusplus
}
#endif
