from melaec3 import InputDeviceConfig, OutputDeviceConfig
import json
import sys
import melaec3
from pathlib import Path
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
from config_loader import AudioSystemConfig

# these are hardcoded to make aec3 happy
TARGET_RATE = 16000
DEFAULT_FRAME = 160
DEFAULT_FILTER = 1600

# We could change these, but no need to
root = Path(__file__).parent
STATE_FILE = root / "last_devices.json"



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
            self.state = self.config.state
            self.output_devices = {}
            self.processing_queue_send = asyncio.Queue(maxsize=0)
            self.processing_queue_recieve = asyncio.Queue(maxsize=0)
            self.callbacks = AsyncCallbackManager()
            self.processing_task = asyncio.create_task(self.audio_processing_task())
            await self.load_cached_config()
        except Exception as e:
            print("Failed to initialize audio system")
            print(traceback.print_exc())
        return self

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
            del self.processing_queue_recieve
            del self.processing_task
            del self.callbacks

    async def calibrate(self):
        await self.processing_queue_send.put((ProcessingEvent.CALIBRATE, None))
        await self.processing_queue_recieve.get()

    async def add_callback(self, callback):
        await self.callbacks.add_callback(callback)
    
    async def remove_callback(self, callback):
        await self.callbacks.remove_callback(callback)
    
    async def audio_processing_task(self):
        try:
            while True:
                while not self.processing_queue_send.empty():
                    processing_message, processing_message_data = self.processing_queue_send.get_nowait()
                    self.processing_queue_send.task_done() # weird stuff processing queue wants
                    if processing_message is not None:
                        match processing_message:
                            case ProcessingEvent.CALIBRATE:
                                await self.stream.calibrate(list(self.output_devices.values()), False)
                                # let it know we finished calibration
                                await self.processing_queue_recieve.put(ProcessingEvent.CALIBRATE_DONE)
                            case _:
                                print(f"Unknown processing message {processing_message}")
                input_data, output_data, aec_data, _, _, vad = await self.stream.update_debug_vad()  
                await self.callbacks(input_data, output_data, aec_data, vad)
                await asyncio.sleep(0) # important to give caller chance to call because above things are hungry
        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception:
            print("Error in audio system")
            print(traceback.print_exc())
        finally:
            pass

    async def load_cached_config(self):
        inputs = self.state.input_devices
        for input_device_config in self.state.input_devices:
            # remove so we can re add it
            if input_device_config in self.state.input_devices:
                self.state.input_devices.remove(input_device_config)
            await self.add_input_device(input_device_config)
        for output_device_config in self.state.output_devices:
            # remove so we can re add it
            if output_device_config in self.state.output_devices:
                self.state.output_devices.remove(output_device_config)
            await self.add_output_device(output_device_config)

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

    async def add_input_device(self, device_config):
        if device_config not in self.state.input_devices:
            await self.stream.add_input_device(device_config.clone_config())
            self.state.input_devices.append(device_config.clone_config())
            self.state.persist_data()
    
    async def remove_input_device(self, device_config):
        if device_config in self.state.input_devices:
            await self.stream.remove_input_device(device_config.clone_config())
            self.state.input_devices.remove(device_config.clone_config())
            self.state.persist_data()
    
    async def add_output_device(self, device_config):
        if device_config not in self.state.output_devices:
            output_device = await self.stream.add_output_device(device_config.clone_config())
            self.output_devices[device_config.clone_config()] = output_device
            self.state.output_devices.append(device_config.clone_config())
            self.state.persist_data()
    
    async def remove_output_device(self, device_config):
        if device_config in self.state.output_devices:
            await self.stream.remove_output_device(device_config.clone_config())
            if device_config in self.output_devices:
                del self.output_devices[device_config.clone_config()]
            self.state.output_devices.remove(device_config.clone_config())
            self.state.persist_data()
    
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
        