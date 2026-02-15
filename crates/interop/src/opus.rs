use core::str;
use std::ffi;
use std::fmt;
use std::fmt::Debug;

mod inner {
    include!(concat!(env!("OUT_DIR"), "/opus_bindings.rs"));
}

pub fn get_version_string() -> Result<String, str::Utf8Error> {
    let s = unsafe { ffi::CStr::from_ptr(inner::_opus_get_version_string()) };
    s.to_str().map(|x| x.to_string())
}

fn opus_strerr(err: i32) -> Result<String, str::Utf8Error> {
    let s = unsafe { ffi::CStr::from_ptr(inner::_opus_strerror(err)) };
    s.to_str().map(|x| x.to_string())
}

#[derive(Debug)]
pub enum SampleRate {
    Hz8000,
    Hz12000,
    Hz16000,
    Hz24000,
    Hz48000,
}

impl SampleRate {
    pub fn as_int32(&self) -> i32 {
        match self {
            SampleRate::Hz8000 => 8000,
            SampleRate::Hz12000 => 12000,
            SampleRate::Hz16000 => 16000,
            SampleRate::Hz24000 => 24000,
            SampleRate::Hz48000 => 48000,
        }
    }
}

#[derive(Debug)]
pub enum Channels {
    Mono,
    Stereo,
}

impl Channels {
    pub fn as_int32(&self) -> i32 {
        match self {
            Channels::Mono => 1,
            Channels::Stereo => 2,
        }
    }
}

#[derive(Debug)]
pub enum Application {
    Voip,
    Audio,
    LowDelay,
}

impl Application {
    pub fn as_int32(&self) -> i32 {
        unsafe {
            match self {
                Application::Voip => inner::_opus_application_voip(),
                Application::Audio => inner::_opus_application_audio(),
                Application::LowDelay => inner::_opus_application_lowdelay(),
            }
        }
    }
}

#[derive(Debug)]
pub enum Bitrate {
    BitsPerSecond(i32),
    Max,
    Auto,
}

impl Bitrate {
    pub fn as_int32(&self) -> i32 {
        unsafe {
            match self {
                Bitrate::BitsPerSecond(x) => *x,
                Bitrate::Max => inner::_opus_bitrate_max(),
                Bitrate::Auto => inner::_opus_bitrate_auto(),
            }
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug)]
pub enum OpusError {
    OPUS_OK,
    OPUS_BAD_ARG,
    OPUS_BUFFER_TOO_SMALL,
    OPUS_INTERNAL_ERROR,
    OPUS_INVALID_PACKET,
    OPUS_UNIMPLEMENTED,
    OPUS_INVALID_STATE,
    OPUS_ALLOC_FAIL,
    UNKNOWN(i32),
}

impl From<i32> for OpusError {
    fn from(value: i32) -> Self {
        match value {
            0 => OpusError::OPUS_OK,
            -1 => OpusError::OPUS_BAD_ARG,
            -2 => OpusError::OPUS_BUFFER_TOO_SMALL,
            -3 => OpusError::OPUS_INTERNAL_ERROR,
            -4 => OpusError::OPUS_INVALID_PACKET,
            -5 => OpusError::OPUS_UNIMPLEMENTED,
            -6 => OpusError::OPUS_INVALID_STATE,
            -7 => OpusError::OPUS_ALLOC_FAIL,
            x => OpusError::UNKNOWN(x),
        }
    }
}

impl From<&OpusError> for i32 {
    fn from(val: &OpusError) -> Self {
        match val {
            OpusError::OPUS_OK => 0,
            OpusError::OPUS_BAD_ARG => -1,
            OpusError::OPUS_BUFFER_TOO_SMALL => -2,
            OpusError::OPUS_INTERNAL_ERROR => -3,
            OpusError::OPUS_INVALID_PACKET => -4,
            OpusError::OPUS_UNIMPLEMENTED => -5,
            OpusError::OPUS_INVALID_STATE => -6,
            OpusError::OPUS_ALLOC_FAIL => -7,
            OpusError::UNKNOWN(x) => *x,
        }
    }
}

impl fmt::Display for OpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}: '{}'",
            self,
            opus_strerr(self.into()).unwrap_or_default()
        )
    }
}

impl std::error::Error for OpusError {}

#[derive(Debug)]
pub struct Encoder {
    ptr: std::ptr::NonNull<::std::os::raw::c_void>,
    channels: usize,
}

impl Encoder {
    pub fn new(
        sample_rate: SampleRate,
        channels: Channels,
        application: Application,
    ) -> Result<Encoder, OpusError> {
        let mut err = 0;
        let p_enc = unsafe {
            inner::_opus_encoder_create(
                sample_rate.as_int32(),
                channels.as_int32(),
                application.as_int32(),
                &mut err,
            )
        };

        if err == 0 {
            Ok(Encoder {
                ptr: unsafe { std::ptr::NonNull::new_unchecked(p_enc) },
                channels: channels.as_int32() as usize,
            })
        } else {
            Err(OpusError::from(err))
        }
    }

    pub fn set_bitrate(&mut self, bitrate: Bitrate) -> Result<(), OpusError> {
        let rc: i32 = unsafe { inner::_opus_set_bitrate(self.ptr.as_ptr(), bitrate.as_int32()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(rc.into())
        }
    }

    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), OpusError> {
        let rc: i32 = unsafe { inner::_opus_set_complexity(self.ptr.as_ptr(), complexity) };
        if rc == 0 {
            Ok(())
        } else {
            Err(rc.into())
        }
    }

    pub fn encode(&self, input: &[i16], output: &mut [u8]) -> Result<usize, OpusError> {
        let rc: i32 = unsafe {
            inner::_opus_encode_i16(
                self.ptr.as_ptr(),
                input.as_ptr(),
                input.len() / self.channels,
                output.as_mut_ptr(),
                output.len(),
            )
        };
        if rc >= 0 {
            Ok(rc as usize)
        } else {
            Err(rc.into())
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe {
            inner::_opus_encoder_destroy(self.ptr.as_ptr());
        }
    }
}

unsafe impl Send for Encoder {}
