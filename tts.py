
from async_callback_manager import AsyncCallbackManager


@dataclass
class AlignmentData:
    start_time_played: float
    # {'chars': [
    # 'W', 'h', 'a', 't', ' ', 'd', 'o', ' ', 'y', 'o', 'u', ' ',
    #  'k', 'n', 'o', 'w', ' ', 'a', 'b', 'o', 'u', 't', ' ', 'w', 'h', 'a', 't',
    # ' ', 'I', "'", 'm', ' ', 'd', 'o', 'i', 'n', 'g', '?', ' '],
    # 'charStartTimesMs':
    # [0, 81, 128, 151, 174, 197, 221, 244, 279, 302, 325, 360, 418, 464, 499, 534, 569, 627, 673, 720, 766, 801, 836, 871, 894, 929, 964, 987, 1022, 1057, 1091, 1138, 1196, 1242, 1358, 1428, 1463, 1533, 1614], 'charDurationsMs': [81, 47, 23, 23, 23, 24, 23, 35, 23, 23, 35, 58, 46, 35, 35, 35, 58, 46, 47, 46, 35, 35, 35, 23, 35, 35, 23, 35, 35, 34, 47, 58, 46, 116, 70, 35, 70, 81, 244]}
    chars: list
    chars_start_times_ms: list

class AsyncTTSStreamer:
    """Async TTS Streamer with interruption capabilities and spoken content tracking."""
    
    def __init__(self, audio_system, config: TTSConfig):
        self.audio_system = audio_system
        self.config = config
        self.speak_task = None
        self.generated_text = ""
        self.word_callback = AsyncCallbackManager()    

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.interrupt()
    
    async def add_callback(self, callback):
        await self.word_callback.add_callback(callback)

    async def remove_callback(self, callback):
        await self.word_callback.remove_callback(callback)

    async def _emit_word_helper(self):
        try:
            prev_offset = 0
            appended_text = ""
            while True:
                offset_in_original_text = self.get_current_index_in_text()
                if prev_offset < offset_in_original_text:
                    match = re.search(r"\w+\b", self.generated_text[prev_offset:])
                    if match:
                        new_text = self.generated_text[prev_offset:prev_offset+match.end()]
                        prev_offset = prev_offset+match.end()
                    else:
                        prev_offset = len(self.generated_text)
                        new_text = self.generated_text[prev_offset:]
                    for word in re.sub(r"\s+", " ", new_text).split(" "):
                        if len(word.strip()) > 0:
                            self.word_callback(word)
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(traceback.print_exc())
            print("Error in osc emit")

    async def _speak_text_helper(self, text_generator: AsyncGenerator[str, None], output_device, output_channel, first_audio_callback) -> bool:
        """
        Speak the given text (task that can be canceled)
        
        Args:
            text_generator: yields text to convert to speech
            
        Returns:
            bool: True if completed successfully, False if interrupted
        """
        current_voice = None
        websocket = None
        self.alignments = []
        self.generated_text = ""
        max_audio_seconds = 60*30
        start_time_played = None
        output_stream = None
        emit_word_task = None
        try:
            emit_word_task = asyncio.create_task(self._emit_word_helper())
            # map first channel of voice to target output channel
            channel_map = {0: [output_channel]}
            output_stream = audio_system.begin_audio_stream(
                output_device,
                1,
                channel_map,
                math.ceil(max_audio_seconds),
                self.config.sample_rate,
                self.config.resampler_quality)
            audios = []
            async def flush_websocket(websocket):
                nonlocal first_audio_callback, start_time_played, output_stream
                if websocket is not None:
                    await websocket.send(json.dumps({
                        "text": "", # final message
                    }))

                    audio_datas = []
                    segment_alignments = []

                    # Get the audio and add it to audio queue
                    # we buffer all audio before playing because that prevents jitters
                    # (this is okay because we do this one sentence at a time)
                    async for message in websocket:
                        data = json.loads(message)
                        audio_base64 = data.get("audio")
                        if audio_base64 and len(audio_base64) > 0:
                            audio_data = base64.b64decode(audio_base64)
                            # call callback (this will stop the "thinking" sound)
                            if not first_audio_callback is None:
                                await first_audio_callback()
                                first_audio_callback = None
                            if start_time_played is None:
                                start_time_played = time.time()

                            float_audio = int16_bytes_to_float(audio_data)
                            if len(float_audio) > 0:
                                audio_datas.append(float_audio)
                                segment_alignments.append((len(float_audio), data["alignment"]))
                        elif data.get("isFinal"):
                            break # done
                    
                    # now send the audio, all in one piece
                    concat_data = np.concatenate(audio_datas)

                    # see when it'll actually be played
                    current_time = time.time()
                    buffered_duration = output_stream.num_queued_samples / self.config.sample_rate
                    output_stream.queue_audio(concat_data)
                    play_start_time = current_time + buffered_duration
                    for buffer_len, alignment in segment_alignments:
                        if alignment is not None: # sometimes we get no alignments but still audio data
                            alignment_data = AlignmentData(
                                start_time_played=play_start_time,
                                chars=alignment['chars'],
                                chars_start_times_ms=alignment['charStartTimesMs']
                            )
                            self.alignments.append(alignment_data)
                        play_start_time += buffer_len/float(shared_sample_rate())


                    await websocket.close()
                    
            async for sentence in stream_sentences(text_generator):
                # add spaces back between the sentences
                self.generated_text = (self.generated_text + " " + sentence).strip()

                # split out *emotive* into seperate parts
                eleven_messages = []
                for text_part, is_emotive in self.extract_emotive_text(sentence):

                    if text_part.strip():
                        voice_id = self.config.emotive_voice_id if is_emotive else self.config.voice_id
                        voice_settings = self._get_voice_settings(is_emotive)

                        eleven_messages.append((voice_id, {
                            "text": text_part.strip(),
                            "voice_settings": voice_settings,
                        }))

                # send text to websockets and receive and play audio
                for voice_id, message in eleven_messages:
                    # Connect to WebSocket (if needed)
                    if current_voice != voice_id:
                        # Disconnect old websocket
                        if not websocket is None:
                            await flush_websocket(websocket)
                            websocket = None
                        current_voice = voice_id
            
                        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
                        params = f"?model_id={self.config.model_id}&output_format={self.config.output_format}"

                        websocket = await websockets.connect(uri + params)

                        # Send initial configuration
                        initial_message = {
                            "text": " ",
                            "voice_settings": message["voice_settings"],
                            "xi_api_key": self.config.api_key
                        }
                        await websocket.send(json.dumps(initial_message))
                    await websocket.send(json.dumps({
                        "text": message['text'].strip() + " ", # elevenlabs always requires ends with single space
                    }))

                # stream the speech, this allows us to start outputting speech before it's done outputting
                await flush_websocket(websocket)
                websocket = None
                current_voice = None

            # wait for buffered audio to drain (polling)
            while output_stream.num_queued_samples > 0:
                await asyncio.sleep(0.05)
                
        except asyncio.CancelledError:
            print(f"Cancelled tts, interrupting playback")
            # not enough audio played and interrupted, make empty
            if start_time_played is None or time.time() - start_time_played < 2.0:
                self.generated_text = ""
            # no audio played yet, empty text
            elif len(self.alignments) == 0:
                self.generated_text = ""
            else:
                # AI Alignment TM
                # (computes where in the text it was interrupted and trims context to that)
                offset_in_original_text = self.get_current_index_in_text()
                self.generated_text = self.generated_text[:offset_in_original_text]
                #print(f"Fuzzy matched to position {offset_in_original_text}")
                #print(self.generated_text)

            # interrupt audio, this clears the buffers
            await interrupt_playback()
            raise
        except Exception as e:
            print(f"TTS error")
            print(traceback.print_exc())
        finally:
            if not output_stream is None:
                self.audio_system.end_audio_stream(output_device, output_stream)
            if emit_word_task:
                try:
                    emit_word_task.cancel()
                    await emit_word_task
                except asyncio.CancelledError:
                    pass # intentional
            # close websocket
            if websocket:
                try:
                    await websocket.close()
                except Exception as e:
                    print(f"TTS websocket close error")
                    print(traceback.print_exc())
            
    async def speak_text(self, text_generator: AsyncGenerator[str, None], first_audio_callback, osc_client, osc_client_address):
        # interrupt (if already running)
        await self.interrupt()

        # start the task (this way it's cancellable and we don't need to spam checks)
        self.speak_task = asyncio.create_task(self._speak_text_helper(text_generator, first_audio_callback, osc_client, osc_client_address))

        # wait for it to finish
        await self.speak_task

        self.speak_task = None
    
    def is_currently_playing(self):
        return self.speak_task is not None
    
    async def interrupt(self):
        if self.speak_task is not None: # if existing one, stop it
            try:
                self.speak_task.cancel()
                await self.speak_task # wait for it to cancel
            except asyncio.CancelledError:
                pass # intentional
            except Exception as e:
                print(f"TTS await error")
                print(traceback.print_exc()) 
            finally:
                self.speak_task = None
        return self.generated_text
    
    async def cleanup(self):
        await self.interrupt()

    def extract_emotive_text(self, text: str) -> List[Tuple[str, bool]]:
        """
        Parse text to separate regular text from emotive text (in asterisks).
        
        Args:
            text: Input text that may contain *emotive* parts
            
        Returns:
            List of (text_chunk, is_emotive) tuples
        """
        if not self.config.emotive_voice_id:
            # No emotive voice configured, return all as regular text
            return [(text, False)]
        
        parts = []
        current_pos = 0
        
        # Find all *text* patterns
        for match in re.finditer(r'\*([^*]+)\*', text):
            # Add regular text before the emotive part
            if match.start() > current_pos:
                regular_text = text[current_pos:match.start()]
                if regular_text.strip():
                    parts.append((regular_text, False))
            
            # Add emotive text (content inside asterisks)
            emotive_text = match.group(1)
            if emotive_text.strip():
                parts.append((emotive_text, True))
            
            current_pos = match.end()
        
        # Add remaining regular text
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            if remaining_text.strip():
                parts.append((remaining_text, False))
        
        return parts if parts else [(text, False)]




_SENTENCE_END = re.compile(r'([.?!][\'")\]]*)(?=\s)')

async def stream_sentences(
    text_stream: AsyncGenerator[str, None]
) -> AsyncGenerator[str, None]:
    buffer = ""

    async for chunk in text_stream:
        buffer += chunk

        while True:
            match = _SENTENCE_END.search(buffer)
            if not match:
                break

            end = match.end(1)
            sentence = buffer[:end].strip()
            buffer = buffer[end:].lstrip()

            if sentence:
                yield sentence

    tail = buffer.strip()
    if tail:
        yield tail