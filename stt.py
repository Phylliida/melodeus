
from deepgram import AsyncDeepgramClient
from deepgram.extensions.types.sockets import (
    ListenV2ConnectedEvent,
    ListenV2FatalErrorEvent,
    ListenV2TurnInfoEvent,
)
import traceback


def float_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 samples in [-1, 1] to signed 16-bit PCM bytes."""
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()

class AsyncSTT(object):
    def __init__(self, deepgram_api_key, audio_system):
        self.deepgram_api_key = deepgram_api_key
        self.audio_system = audio_system

    async def __aenter__(self):
        self.prev_turn_idx = None
        self.prev_transcript = ""
        self.prev_audio_window_start = 0
        self.prev_audio_window_end = 0
        self.current_turn_history = []
        self.current_turn_autosent_transcript = None
        self.last_sent_message_uuid = None
        self.deepgram_task = asyncio.create_task(self.deepgram_processor())
        

    async def __aexit__(self, exc_type, exc, tb):
        self.deepgram_task.cancel()
        try:
            await self.deepgram_task
        except asyncio.CancelledError:
            pass # intentional
        except:
            print(f"Error in deepgram task cancel")
            print(traceback.print_exc())
        del self.deepgram_task

    async def deepgram_processor(self):
        while True:
            try:
                deepgram = AsyncDeepgramClient(api_key=self.deepgram_api_key)
                async with self.deepgram.listen.v2.connect(
                    model="flux-general-en",
                    encoding="linear16",
                    sample_rate="16000",
                    eot_timeout_ms="500",
                    keyterm=params['keyterm']) as connection:
                    connection.on(EventType.OPEN, self.deepgram_on_open)
                    connection.on(EventType.MESSAGE, self.deepgram_on_message)
                    connection.on(EventType.ERROR, self.deepgram_error)
                    connection.on(EventType.CLOSE, self.deepgram_close)
                    def audio_callback(audio_input_data, audio_output_data, aec_input_data, channel_vads):
                        # if we have any input channels available and aec detected something, mix them
                        # alternatively we could run deepgram per channel but that would be more expensive
                        if len(aec_input_data) > 0 and len(aec_input_data[0]) > 0 and any(channel_vads):
                            mixed_data = np.zeros(len(aec_input_data[0]))
                            for i in range(len(channel_vads)):
                                vad = channel_vads[i]
                                if vad:
                                    mixed_data += np.array(aec_input_data[i])
                            mixed_data_pcm = float_to_int16_bytes(mixed_data)
                            if self.voice_fingerprinter is not None:
                                self.voice_fingerprinter.add_audio_chunk(
                                    mixed_data,
                                    self.num_audio_frames_recieved,
                                    self.config.sample_rate,
                                )
                                # increment frame count
                                self.num_audio_frames_recieved += len(audio_np)
                            
                            await connection.send_media(mixed_data_pcm)
                    self.audio_system.add_callback(audio_callback)
                    await connection.start_listening()
                    self.audio_system.remove_callback(audio_callback)
            except asyncio.CancelledError:
                print("Deepgram processs canceled")
                break
            except:
                print("Error in deepgram")
                print(traceback.print_exc())


    def deepgram_on_open(self, *args, **kwargs):
        """Handle Deepgram connection open."""
        print("🔗 Deepgram STT connection opened")

    async def deepgram_on_message(self, message: Any):
        """Dispatch Deepgram websocket messages to the appropriate handlers."""
        try:
            if isinstance(message, ListenV2TurnInfoEvent):
                await self.deepgram_turn(stt_state, message)
            elif isinstance(message, ListenV2FatalErrorEvent):
                print(f"❌ Deepgram fatal error ({message.code}): {message.description}")
            elif isinstance(message, ListenV2ConnectedEvent):
                # Already handled by EventType.OPEN; nothing additional needed.
                pass
        except Exception as dispatch_error:
            print(f"⚠️  Failed to process Deepgram message:")
            print(traceback.print_exc())

    async def deepgram_turn(self, message: ListenV2TurnInfoEvent):
        event_type = (message.event or "").lower()
        turn_idx = message.turn_index
        transcript = message.transcript or ""
        # speech started: todo
        # if event_type == "startofturn":
        #     await self._on_speech_started()
        if transcript.strip() != "":
            print(transcript)

        send_message = False
        edit_message = False
        # code to end turn earlier than deepgram decides for decreased latency
        if turn_idx != self.prev_turn_idx:
            self.current_turn_history = []    
        if len(transcript) > 0:
            self.current_turn_history.append((time.time(), transcript))
        if turn_idx == self.prev_turn_idx and len(transcript) > 0 and self.has_meaningful_change(transcript, self.current_turn_autosent_transcript):
            ms_until_autosend = 750
            # if we took longer than that and still the same, autosend
            oldest_time_still_same = time.time()
            for old_time, old_transcript in self.current_turn_history[::-1]:
                if old_transcript != transcript:
                    break
                else:
                    oldest_time_still_same = old_time
            if (time.time() - oldest_time_still_same)*1000 > ms_until_autosend:
                #print("Autosend")
                if self.current_turn_autosent_transcript is not None:
                    edit_message = True
                else:
                    send_message = True
                self.current_turn_autosent_transcript = transcript
            
        # If we are on a new turn, send old turn
        if turn_idx != self.prev_turn_idx and self.prev_turn_idx != None and self.prev_transcript:
            if self.current_turn_autosent_transcript is not None:
                if self.has_meaningful_change(self.current_turn_autosent_transcript, self.prev_transcript):
                    edit_message = True
            else:
                send_message = True
            
        if send_message or edit_message:
            if self.voice_fingerprinter:
                word_timing = WordTiming(
                    word=self.prev_transcript,
                    speaker_id=f"speaker {turn_idx}", # v2 doesn't have diarization yet
                    start_time=self.prev_audio_window_start,
                    end_time=self.prev_audio_window_end,
                    confidence=1)
                user_tag = self.voice_fingerprinter.process_transcript_words(word_timings=[word_timing], sample_rate=self.config.sample_rate)
            else:
                user_tag = "User"
            message_uuid = str(uuid.uuid4()) if send_message else self.last_sent_message_uuid
            self.last_sent_message_uuid = message_uuid
            #print(("send" if send_message else "edit"), "with data")
            #print(self.prev_transcript)
            stt_result = STTResult(
                text = self.prev_transcript,
                confidence = 1.0,
                is_final = True,
                is_edit = edit_message,
                speaker_id = None,
                speaker_name = user_tag,
                timestamp = datetime.now(),
                message_id = message_uuid,
            )
            await self._emit_event(
                STTEventType.UTTERANCE_COMPLETE,
                stt_result
            )
        if turn_idx != self.prev_turn_idx:
            self.last_sent_message_uuid = None
            self.current_turn_autosent_transcript = None
            

        self.prev_turn_idx = turn_idx
        self.prev_transcript = transcript
        self.prev_audio_window_start = message.audio_window_start
        self.prev_audio_window_end = message.audio_window_end

        if transcript and not send_message and not edit_message and self.last_sent_message_uuid is None:
            self.most_recent_turn_text = transcript
            
            stt_result = STTResult(
                text = transcript,
                confidence = 1.0,
                is_final = False,
                speaker_id = None,
                speaker_name =  None,
                timestamp = datetime.now(),
                message_id = None,
                is_edit = False
            ) 

            await self._emit_event(
                STTEventType.INTERIM_RESULT,
                stt_result
            )
            print(f"Turn {turn_idx} {transcript}")
        else:
            #print(f"Turn {turn_idx} Dummy turn")
            pass
        #if event_type in {"turn_end", "speech_ended", "speech_end"}:
        #    self._on_utterance_end(message)


    async def deepgram_error(self, error: Exception):
        """Handle errors emitted by the Deepgram websocket client."""
        error_message = str(error)
        print(f"⚠️ Deepgram websocket error")
        print(error_message)



            