
from deepgram import AsyncDeepgramClient
from deepgram.extensions.types.sockets import (
    ListenV2ConnectedEvent,
    ListenV2FatalErrorEvent,
    ListenV2TurnInfoEvent,
)
import traceback

class AsyncSTT(object):

    async def __aenter__(self):
        self.prev_turn_idx = None
        self.prev_transcript = ""
        self.prev_audio_window_start = 0
        self.prev_audio_window_end = 0
        self.current_turn_history = []
        self.current_turn_autosent_transcript = None
        self.last_sent_message_uuid = None

    async def __aexit__(self, exc_type, exc, tb):
        

    async def initialize_deepgram(self, api_key):
        deepgram = AsyncDeepgramClient(api_key=config.api_key)
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
            await connection.start_listening()


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



            