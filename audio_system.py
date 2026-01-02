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

# these are hardcoded to make aec3 happy
TARGET_RATE = 16000
DEFAULT_FRAME = 160
DEFAULT_FILTER = 1600

# We could change these, but no need to
DEFAULT_OUTPUT_FRAME = 16
HISTORY_LEN = 100
CALIBRATION_PACKETS = 20
AUDIO_BUF_SECONDS = 20
RESAMPLE_QUALITY = 5


root = Path(__file__).parent
STATE_FILE = root / "last_devices.json"


@dataclass(slots=True)
class AudioSystemState:
    input_devices: List[dict] = field(default_factory=list)
    output_devices: List[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return {
            'input_devices': [cfg.to_dict() for cfg in self.input_devices],
            'output_devices': [cfg.to_dict() for cfg in self.output_devices]
        }

    @classmethod
    def from_json(cls, raw: str) -> "AudioSystemState":
        res = cls(**json.loads(raw))
        res.input_devices = [InputDeviceConfig.from_dict(cfg) for cfg in res.input_devices]
        res.output_devices = [OutputDeviceConfig.from_dict(cfg) for cfg in res.output_devices]
        return res

class ProcessingEvent(StrEnum):
    CALIBRATE = "calibrate"
    CALIBRATE_DONE = "calibrate done"
    ADD_CALLBACK = "add callback"
    REMOVE_CALLBACK = "remove callback"

class AudioSystem(object):
    
    async def __aenter__(self):
        try:
            stream_cfg = melaec3.AecConfig(target_sample_rate=TARGET_RATE, frame_size=DEFAULT_FRAME, filter_length=DEFAULT_FILTER)
            self.stream = melaec3.AecStream(stream_cfg)
            self.state = AudioSystemState()
            self.output_devices = {}
            self.processing_queue_send = asyncio.Queue(maxsize=0)
            self.processing_queue_recieve = asyncio.Queue(maxsize=0)
            self.processing_task = asyncio.create_task(self.audio_processing_task())
            self.callbacks = []
        except Exception as e:
            print("Failed to pre-init aec")
            print(traceback.print_exc())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.processing_task.cancel()
        try:
            await self.processing_task
        except asyncio.CancelledError:
            pass # intentional
        except:
            print(f"Error in audio system task cancel")
            print(traceback.print_exc())
        del self.output_devices
        del self.state
        del self.stream
        del self.processing_queue_send
        del self.processing_queue_recieve
        del self.processing_task
        del self.callbacks


    async def calibrate(self):
        # This has a subtle bug where later calls can finish before earlier ones
        # We'll allow that, but do keep it in mind
        await self.processing_queue_send.put((ProcessingEvent.CALIBRATE, None))
        await self.processing_queue_recieve.recieve()

    def add_callback(self, callback):
        self.processing_queue_send.put((ProcessingEvent.ADD_CALLBACK, callback))
    
    def remove_callback(self, callback):
        self.processing_queue_send.put((ProcessingEvent.REMOVE_CALLBACK, callback))
    
    async def audio_processing_task(self):
        try:
            while True:
                processing_message = None
                processing_message_data = None
                try:
                    processing_message, processing_message_data = self.processing_queue_send.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                else:
                    self.processing_queue_send.task_done() # weird stuff processing queue wants
                if processing_message is not None:
                    match processing_message:
                        case ProcessingEvent.CALIBRATE:
                            await self.stream.calibrate(list(self.output_devices.values()), False)
                            # let it know we finished calibration
                            self.processing_queue_recieve.put(ProcessingEvent.CALIBRATE_DONE)
                        case ProcessingEvent.ADD_CALLBACK:
                            self.callbacks.append(processing_message_data)
                        case ProcessingEvent.REMOVE_CALLBACK:
                            if processing_message_data in self.callbacks:
                                self.callbacks.remove(processing_message_data)
                        case _:
                            print(f"Unknown processing message {processing_message}")
                input_data, output_data, aec_data, _, _, vad = await self.stream.update_debug_vad()  
                for callback in self.callbacks:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(input_data, output_data, aec_data, vad)
                    else:
                        callback(input_data, output_data, aec_data, vad)
        except asyncio.CancelledError:
          raise
        except Exception as e:
            print("Error in audio system")
            print(traceback.print_exc())
        finally:
            pass

    def load_cached_config(self):
        self.state = AudioSystemState()
        try:
            json_data = jsonlib.loads(STATE_FILE.read_text())
            self.state = AudioSystemState.from_json(json_data)
        except Exception:
            pass
        for input_device_config in self.state.input_devices:
            self.add_input_device(input_device_config)
        for output_device_config in self.state.output_devices:
            self.add_output_device(output_device_config)
    
    def write_cached_config(self):
        try:
            STATE_FILE.write_text(jsonlib.dumps(self.state.to_json()))
        except Exception as e:
            print("failed to write state", e)

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
            self.write_cached_config()
    
    async def remove_input_device(self, device_config):
        if device_config in self.state.input_devices:
            await self.stream.remove_input_device(device_config.clone_config())
            self.state.input_devices.remove(device_config.clone_config())
            self.write_cached_config()
    
    async def add_output_device(self, device_config):
        if device_config not in self.state.output_devices:
            output_device = await self.stream.add_output_device(device_config)
            self.output_devices[device_config.clone_config()] = output_device
            self.state.output_devices.append(device_config.clone_config())
            self.write_cached_config()
    
    async def remove_output_device(self, device_config):
        if device_config in self.state.output_devices:
            await self.stream.remove_output_device(device_config.clone_config())
            self.state.output_devices.remove(device_config.clone_config())
            self.write_cached_config()
    
    def get_connected_output_devices(self):
        return [config.clone_config() for config in self.state.output_devices.keys()]

    def begin_audio_stream(self, output_device, channels, channel_map, audio_buffer_seconds, sample_rate, resampler_quality):
        return self.output_devices[output_device].begin_audio_stream(
            channels,
            channel_map,
            audio_buffer_seconds,
            sample_rate,
            resampler_quality
        )

    def end_audio_stream(self, output_device, stream):
        self.output_devices[output_device].end_audio_stream(stream)
        