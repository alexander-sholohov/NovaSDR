#include "opus_wrapper.h"
#include <opus/opus.h>

int32_t _opus_application_audio()
{
    return OPUS_APPLICATION_AUDIO;
}

int32_t _opus_application_voip()
{
    return OPUS_APPLICATION_VOIP;
}

int32_t _opus_application_lowdelay()
{
    return OPUS_APPLICATION_RESTRICTED_LOWDELAY;
}

int32_t _opus_bitrate_max()
{
    return OPUS_BITRATE_MAX;
}

int32_t _opus_bitrate_auto()
{
    return OPUS_AUTO;
}

void *_opus_encoder_create(int32_t fs, int32_t channels, int32_t application, int32_t *err)
{
    // allowed fs: 48000, 24000, 16000, 12000, 8000
    // allowed channels: 1, 2
    int local_err = 0;
    void *res = opus_encoder_create(fs, channels, application, &local_err);
    *err = local_err; // safe translation C-int to Rust's i32
    return res;
}

void _opus_encoder_destroy(void *enc)
{
    opus_encoder_destroy(enc);
}

int32_t _opus_set_bitrate(void *enc, int32_t bitrate)
{
    // Rates from 500 to 512000 bits per second are meaningful
    return opus_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate));
}

int32_t _opus_set_complexity(void *enc, int32_t complexity)
{
    return opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(complexity));
}

int32_t _opus_encode_i16(void *enc, const int16_t *pcm, size_t frame_size, uint8_t *data, size_t max_data_bytes)
{
    // To encode a frame, opus_encode() or opus_encode_float() must be called with exactly one frame (2.5, 5, 10, 20, 40 or 60 ms) of audio data:
    return opus_encode(enc, pcm, frame_size, data, max_data_bytes);
}

int32_t _opus_encode_float(void *enc, const float *pcm, size_t frame_size, uint8_t *data, size_t max_data_bytes)
{
    // To encode a frame, opus_encode() or opus_encode_float() must be called with exactly one frame (2.5, 5, 10, 20, 40 or 60 ms) of audio data:
    return opus_encode_float(enc, pcm, frame_size, data, max_data_bytes);
}

void *_opus_decoder_create(int32_t fs, int32_t channels, int32_t *error)
{
    // allowed fs: 48000, 24000, 16000, 12000, 8000
    // allowed channels: 1, 2
    int local_err = 0;
    void *res = opus_decoder_create(fs, channels, &local_err);
    *error = local_err;
    return res;
}

int32_t _opus_decode_i16(void *dec, const uint8_t *data, size_t len, int16_t *pcm, size_t frame_size, int32_t decode_fec)
{
    // decode_fec: Flag (0 or 1) to request that any in-band forward error correction data be
    return opus_decode(dec, data, len, pcm, frame_size, decode_fec);
}

int32_t _opus_decode_float(void *dec, const uint8_t *data, size_t len, float *pcm, size_t frame_size, int32_t decode_fec)
{
    return opus_decode_float(dec, data, len, pcm, frame_size, decode_fec);
}

void _opus_decoder_destroy(void *dec)
{
    opus_decoder_destroy(dec);
}

const char *_opus_get_version_string(void)
{
    return opus_get_version_string();
}

const char *_opus_strerror(int32_t error)
{
    return opus_strerror(error);
}
