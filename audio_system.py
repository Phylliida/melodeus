from melaec3 import InputDeviceConfig, OutputDeviceConfig
import json
import sys
import melaec3
from pathlib import Path
import numpy as np
import wave
import contextlib
from typing import List
from dataclasses import dataclass, field
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum
import json as jsonlib
import traceback
import asyncio
from async_callback_manager import AsyncCallbackManager
from config_loader import AudioSystemConfig, AudioSystemState

# Defaults pulled from pure-rust-aec examples
HISTORY_LEN = 100
CALIBRATION_PACKETS = 20
AUDIO_BUF_SECONDS = 20
RESAMPLE_QUALITY = 5
FRAME_SIZE_MILLIS = 3
AEC_SAMPLE_RATE = 16000

MAX_CALIBRATION_OFFSET_FRAMES = 1000
MAX_CALIBRATION_RETRY_ATTEMPTS = 3

# these are hardcoded to make aec3 happy
TARGET_RATE = AEC_SAMPLE_RATE
DEFAULT_FRAME = 160
DEFAULT_FILTER = DEFAULT_FRAME * 10
DEFAULT_OUTPUT_FRAME = 160

# We could change these, but no need to
root = Path(__file__).parent
STATE_FILE = root / "last_devices.json"

def load_wav(path):
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        if w.getsampwidth() != 2:
            raise ValueError("sample must be 16-bit")
        frames = w.getnframes()
        buf = w.readframes(frames)

    audio_seconds = frames / float(rate)
    samples = np.frombuffer(buf, dtype=np.int16).reshape(-1, ch)
    floats = samples.astype(np.float32) / 32768.0  # interleaved [frames, ch]
    interleaved = floats.reshape(-1).astype(np.float32, copy=False)
    return rate, ch, audio_seconds, interleaved

class ProcessingEvent(StrEnum):
    CALIBRATE = "calibrate"
    CALIBRATE_DONE = "calibrate done"

class AudioSystem(object):

    '''
    resume restarts audio system with same devices and settings as before
    '''
    def __init__(self, config: AudioSystemConfig):
        self.config = config
    
    async def __aenter__(self):
        try:
            stream_cfg = melaec3.AecConfig(target_sample_rate=TARGET_RATE, frame_size=DEFAULT_FRAME, filter_length=DEFAULT_FILTER)
            self.stream = melaec3.AecStream(stream_cfg)
            # convert it into the form we need
            print("Setting state")
            self.config.state = AudioSystemState.from_json(self.config.state.to_json())
            print("Done setting state")
            self.state = self.config.state
            self.output_devices = {}
            self.processing_queue_send = asyncio.Queue(maxsize=0)
            self.audio_callbacks = AsyncCallbackManager()
            self.ready_input_device_callbacks = AsyncCallbackManager()
            self.ready_output_device_callbacks = AsyncCallbackManager()
            
            # add auto calibration callbacks once devices are ready (it can take a bit for the device to calibrate due to sync code, which is why we can't call calibrate immediately)
            await self.add_ready_input_device_callback(self.auto_calibrate_callback)
            await self.add_ready_output_device_callback(self.auto_calibrate_callback)

            self.processing_task = asyncio.create_task(self.audio_processing_task())
            await self.load_cached_config()
        except Exception as e:
            print("Failed to initialize audio system")
            print(traceback.print_exc())
        return self

    async def auto_calibrate_callback(self, new_ready_device, all_ready_input_devices, all_ready_output_devices):
        print(f"ready device {new_ready_device}")
        print(f"ready input devices {all_ready_input_devices} ")
        print(f"ready output devices {all_ready_output_devices} ")
        if self.config.auto_calibrate and len(all_ready_input_devices) > 0 and len(all_ready_output_devices) > 0:
            devices = [all_ready_input_devices, all_ready_output_devices]
            # avoid double calibration init
            if not hasattr(self, "devices_last_calibration") or self.devices_last_calibration != devices:
                self.devices_last_calibration = devices
                print("Auto calibration")
                await self.calibrate()
                print("Done auto calibration")

    async def __aexit__(self, exc_type, exc, tb):
        self.processing_task.cancel()
        try:
            await self.processing_task
        except asyncio.CancelledError:
            pass # intentional
        except KeyboardInterrupt:
            raise
        except Exception:
            print(f"Error in audio system task cancel")
            print(traceback.print_exc())
        finally:
            del self.output_devices
            del self.state
            del self.stream
            del self.processing_queue_send
            del self.processing_task
            del self.audio_callbacks
            del self.ready_input_device_callbacks
            del self.ready_output_device_callbacks

    async def calibrate(self, max_offset_frames, max_calibration_attempts):
        # If we’re already on the processing task, run directly to avoid deadlock.
        if asyncio.current_task() is self.processing_task:
            await self.stream.calibrate(max_offset_frames, max_calibration_attempts, list(self.output_devices.values()), False)
        # otherwise, enqueue it then wait for processing task to finish
        else:
            fut = loop.create_future()
            await self.processing_queue_send.put((ProcessingEvent.CALIBRATE, fut))
            await fut # wait for completion
    async def add_audio_callback(self, callback):
        await self.audio_callbacks.add_callback(callback)
    
    async def remove_audio_callback(self, callback):
        await self.audio_callbacks.remove_callback(callback)
    
    async def add_ready_input_device_callback(self, callback):
        await self.ready_input_device_callbacks.add_callback(callback)

    async def remove_ready_input_device_callback(self, callback):
        await self.ready_input_device_callbacks.remove_callback(callback)
    
    async def add_ready_output_device_callback(self, callback):
        await self.ready_output_device_callbacks.add_callback(callback)

    async def remove_ready_output_device_callback(self, callback):
        await self.ready_output_device_callbacks.remove_callback(callback)

    async def audio_processing_task(self):
        try:
            i = 0
            while True:
                while not self.processing_queue_send.empty():
                    processing_message, processing_message_data = self.processing_queue_send.get_nowait()
                    self.processing_queue_send.task_done() # weird stuff processing queue wants
                    if processing_message is not None:
                        match processing_message:
                            case ProcessingEvent.CALIBRATE:
                                await self.stream.calibrate(MAX_CALIBRATION_OFFSET_FRAMES, MAX_CALIBRATION_RETRY_ATTEMPTS, list(self.output_devices.values()), False)
                                if processing_message_data: # trigger that it is done
                                    processing_message_data.set_result(None)
                            case _:
                                print(f"Unknown processing message {processing_message}")
                new_ready_inputs, new_ready_outputs, input_bytes, output_bytes, aec_bytes, _, _, vad = await self.stream.update_debug_vad()  
                
                # callbacks for new devices ready
                if len(new_ready_inputs) > 0:
                    all_ready_inputs = self.stream.get_ready_input_devices()
                    all_ready_outputs = self.stream.get_ready_output_devices()
                    for ready_input in new_ready_inputs:
                        await self.ready_input_device_callbacks(ready_input, all_ready_inputs, all_ready_outputs)
                if len(new_ready_outputs) > 0:
                    all_ready_inputs = self.stream.get_ready_input_devices()
                    all_ready_outputs = self.stream.get_ready_output_devices()
                    for ready_output in new_ready_outputs:
                        await self.ready_output_device_callbacks(ready_output, all_ready_inputs, all_ready_outputs)
                input_data  = np.frombuffer(input_bytes, dtype=np.float32)
                output_data = np.frombuffer(output_bytes, dtype=np.float32)
                aec_data    = np.frombuffer(aec_bytes, dtype=np.float32)
                input_channels = self.stream.num_input_channels
                output_channels = self.stream.num_output_channels
                await self.audio_callbacks(input_channels, output_channels, input_data, output_data, aec_data, vad)
                await asyncio.sleep(0) # important to give caller chance to call because above things are hungry
        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception:
            print("Error in audio system")
            print(traceback.print_exc())
        finally:
            await self._shutdown_streams()
            pass

    async def _shutdown_streams(self):
        """
        Best-effort stop of active devices so CPAL callbacks halt before the stream is dropped.
        Leaves state intact so persisted config is not cleared.
        """
        if not hasattr(self, "stream"):
            return
        # Stop outputs first so any dependent processing halts before inputs.
        for output_cfg in self.state.output_devices:
            print(f"removing output {output_cfg}")
            await self.remove_output_device(output_cfg.clone_config(), update_config=False)
        for input_cfg in self.state.input_devices:
            print(f"removing input {input_cfg}")
            await self.remove_input_device(input_cfg.clone_config(), update_config=False)
        self.output_devices.clear()
    
    async def load_cached_config(self):
        inputs = self.state.input_devices
        outputs = self.state.output_devices
        # clear so we can re add them
        self.state.input_devices = []
        self.state.output_devices = []
        for input_device_config in inputs:
            await self.add_input_device(input_device_config.clone_config(), update_config=False)
        for output_device_config in outputs:
            await self.add_output_device(output_device_config.clone_config(), update_config=False)

    def get_supported_device_configs(self):
        ins = melaec3.get_supported_input_configs(
            history_len=HISTORY_LEN,
            num_calibration_packets=CALIBRATION_PACKETS,
            audio_buffer_seconds=AUDIO_BUF_SECONDS,
            resampler_quality=RESAMPLE_QUALITY,
        )
        outs = melaec3.get_supported_output_configs(
            history_len=HISTORY_LEN,
            num_calibration_packets=CALIBRATION_PACKETS,
            audio_buffer_seconds=AUDIO_BUF_SECONDS,
            resampler_quality=RESAMPLE_QUALITY,
            frame_size=DEFAULT_OUTPUT_FRAME,
        )
        
        return {
            "inputs": ins,
            "outputs": outs
        }

    async def add_input_device(self, device_config, update_config=True):
        if device_config.clone_config() not in self.state.input_devices:
            await self.stream.add_input_device(device_config.clone_config())
            print(f"Adding input device {device_config.to_json()}")
            self.state.input_devices.append(device_config.clone_config())
            if update_config: self.config.persist_data()
    
    async def remove_input_device(self, device_config, update_config=True):
        if device_config.clone_config() in self.state.input_devices:
            await self.stream.remove_input_device(device_config.clone_config())
            print(f"Removing input device {device_config.to_json()}")
            self.state.input_devices.remove(device_config.clone_config())
            if update_config: self.config.persist_data()
    
    async def add_output_device(self, device_config, update_config=True):
        if device_config.clone_config() not in self.state.output_devices:
            output_device = await self.stream.add_output_device(device_config.clone_config())
            print(f"Adding output device {device_config.to_json()}")
            self.output_devices[device_config.clone_config()] = output_device
            self.state.output_devices.append(device_config.clone_config())
            if update_config: self.config.persist_data()
    
    async def remove_output_device(self, device_config, update_config=True):
        if device_config.clone_config() in self.state.output_devices:
            await self.stream.remove_output_device(device_config.clone_config())
            print(f"Removing output device {device_config.to_json()}")
            if device_config.clone_config() in self.output_devices:
                del self.output_devices[device_config.clone_config()]
            self.state.output_devices.remove(device_config.clone_config())
            if update_config: self.config.persist_data()
    
    def get_connected_output_devices(self):
        return [config.clone_config() for config in self.output_devices.keys()]

    async def begin_audio_stream(self, output_device, channels, channel_map, audio_buffer_seconds, sample_rate, resampler_quality):
        return self.output_devices[output_device].begin_audio_stream(
            channels,
            channel_map,
            audio_buffer_seconds,
            sample_rate,
            resampler_quality
        )

    async def interrupt_all_audio_streams(self, output_device):
        self.output_devices[output_device].interrupt_all_streams()

    async def end_audio_stream(self, output_device, stream):
        self.output_devices[output_device].end_audio_stream(stream)
        
