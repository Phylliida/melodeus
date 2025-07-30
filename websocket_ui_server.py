"""WebSocket server for UI communication with voice system."""
import asyncio
import websockets
import json
import time
from typing import Set, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

@dataclass
class UIMessage:
    """Base class for UI messages."""
    type: str
    data: Dict[str, Any]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


class VoiceUIServer:
    """WebSocket server for voice conversation UI."""
    
    def __init__(self, conversation=None, host='localhost', port=8765):
        self.conversation = conversation
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        self.logger = logging.getLogger(__name__)
        
        # Track current state for new clients
        self.current_state = {
            "current_speaker": None,
            "is_speaking": False,
            "is_processing": False,
            "pending_speaker": None,
            "thinking_sound": False,
            "stt_active": True,
            "conversation_active": False
        }
        
    async def start(self):
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self.handle_client, self.host, self.port
        )
        print(f"🌐 WebSocket UI server started on ws://{self.host}:{self.port}")
        
    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("🌐 WebSocket UI server stopped")
            
    async def handle_client(self, websocket):
        """Handle a new client connection."""
        # Register client
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        print(f"🔌 UI client connected from {client_addr}")
        
        try:
            # Send initial state
            await self.send_state_sync(websocket)
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await self.send_error(websocket, "Invalid JSON message")
                except Exception as e:
                    self.logger.error(f"Error handling message: {e}")
                    await self.send_error(websocket, str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print(f"🔌 UI client disconnected from {client_addr}")
            
    async def handle_message(self, websocket, message: Dict[str, Any]):
        """Handle incoming message from UI client."""
        msg_type = message.get("type")
        data = message.get("data", {})
        
        if not self.conversation:
            await self.send_error(websocket, "Voice system not connected")
            return
            
        try:
            if msg_type == "force_interrupt":
                await self.handle_force_interrupt()
                
            elif msg_type == "stt_control":
                action = data.get("action")
                await self.handle_stt_control(action)
                
            elif msg_type == "select_speaker":
                speaker = data.get("speaker")
                await self.handle_select_speaker(speaker)
                
            elif msg_type == "trigger_speaker":
                speaker = data.get("speaker")
                await self.handle_trigger_speaker(speaker)
                
            elif msg_type == "sync_request":
                await self.send_state_sync(websocket)
                
            elif msg_type == "ui_ready":
                # Client is ready, maybe send recent history
                client_id = data.get("client_id", "unknown")
                print(f"✅ UI client ready: {client_id}")
                
            else:
                await self.send_error(websocket, f"Unknown message type: {msg_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling {msg_type}: {e}")
            await self.send_error(websocket, str(e))
            
    async def handle_force_interrupt(self):
        """Handle force interrupt request."""
        if not self.conversation:
            return
            
        print("🛑 Force interrupt requested by UI")
        
        # Stop TTS
        if hasattr(self.conversation, 'tts'):
            await self.conversation.tts.stop()
            
        # Cancel current LLM task
        if hasattr(self.conversation.state, 'current_llm_task'):
            if self.conversation.state.current_llm_task and not self.conversation.state.current_llm_task.done():
                self.conversation.state.current_llm_task.cancel()
                
        # Stop thinking sound
        if hasattr(self.conversation, 'thinking_sound'):
            await self.conversation.thinking_sound.stop()
                
        # Clear processing state
        self.conversation.state.is_processing_llm = False
        self.conversation.state.is_speaking = False
        
        # Increment processing generation to cancel any ongoing processing
        if hasattr(self.conversation, '_processing_generation'):
            self.conversation._processing_generation += 1
            
        # Also increment director generation to cancel any director requests
        if hasattr(self.conversation, '_director_generation'):
            self.conversation._director_generation += 1
            
        # Cancel any pending processing task
        if hasattr(self.conversation, '_processing_task'):
            if self.conversation._processing_task and not self.conversation._processing_task.done():
                self.conversation._processing_task.cancel()
        
        # Update and broadcast state
        self.current_state.update({
            "current_speaker": None,
            "is_speaking": False,
            "is_processing": False,
            "thinking_sound": False
        })
        
        await self.broadcast_speaker_status()
        
    async def handle_stt_control(self, action: str):
        """Handle STT pause/resume."""
        if not self.conversation or not hasattr(self.conversation, 'stt'):
            return
            
        if action == "pause":
            await self.conversation.stt.pause()
            self.current_state["stt_active"] = False
            print("⏸️ STT paused by UI")
        elif action == "resume":
            await self.conversation.stt.resume()
            self.current_state["stt_active"] = True
            print("▶️ STT resumed by UI")
            
        await self.broadcast_system_status()
        
    async def handle_select_speaker(self, speaker: str):
        """Handle manual speaker selection."""
        if not self.conversation:
            return
            
        print(f"🎭 UI requested speaker: {speaker}")
        # TODO: Implement manual speaker selection
        # This would bypass director and force a specific speaker
        
    async def handle_trigger_speaker(self, speaker: str):
        """Handle triggering a specific speaker to respond."""
        if not self.conversation:
            return
            
        print(f"🎯 UI triggered speaker: {speaker}")
        
        # Import necessary classes
        from unified_voice_conversation_config import ConversationTurn
        from datetime import datetime
        
        # Create a conversation turn with metadata
        trigger_turn = ConversationTurn(
            role="user",
            content=f"[System: {speaker}, please share your thoughts]",
            timestamp=datetime.now(),
            status="pending",
            metadata={
                "is_manual_trigger": True,
                "triggered_speaker": speaker
            }
        )
        
        # Add to conversation history
        self.conversation.state.conversation_history.append(trigger_turn)
        
        # Log the turn
        self.conversation._log_conversation_turn(trigger_turn.role, trigger_turn.content)
        
        # Force processing of pending utterances
        asyncio.create_task(self.conversation._process_pending_utterances())
        
        # Broadcast status update
        await self.broadcast_speaker_status(
            is_processing=True,
            pending_speaker=speaker
        )
        
    async def send_state_sync(self, websocket):
        """Send current state to a client."""
        # Send speaker status
        await self.send_to_client(websocket, UIMessage(
            type="speaker_status",
            data=self.current_state
        ))
        
        # Send system status
        await self.send_to_client(websocket, UIMessage(
            type="system_status",
            data={
                "stt_active": self.current_state["stt_active"],
                "tts_active": bool(self.current_state["is_speaking"]),
                "conversation_active": self.current_state["conversation_active"],
                "current_generation": getattr(self.conversation, '_processing_generation', 0) if self.conversation else 0,
                "director_generation": getattr(self.conversation, '_director_generation', 0) if self.conversation else 0
            }
        ))
        
        # Send available characters
        if self.conversation and hasattr(self.conversation, 'character_manager') and self.conversation.character_manager:
            characters = []
            for char_name in self.conversation.character_manager.characters.keys():
                char_config = self.conversation.character_manager.get_character_config(char_name)
                characters.append({
                    "name": char_name,
                    "model": char_config.llm_model if char_config else "unknown",
                    "active": self.conversation.character_manager.active_character == char_name
                })
            
            await self.send_to_client(websocket, UIMessage(
                type="available_characters",
                data={"characters": characters}
            ))
        
    async def send_to_client(self, client: websockets.WebSocketServerProtocol, message: UIMessage):
        """Send message to specific client."""
        try:
            await client.send(message.to_json())
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"Error sending to client: {e}")
            
    async def broadcast(self, message: UIMessage):
        """Broadcast message to all connected clients."""
        if self.clients:
            await asyncio.gather(
                *[self.send_to_client(client, message) for client in self.clients],
                return_exceptions=True
            )
            
    async def send_error(self, client, error_message: str, severity="error"):
        """Send error message to client."""
        await self.send_to_client(client, UIMessage(
            type="error",
            data={
                "severity": severity,
                "message": error_message,
                "timestamp": time.time()
            }
        ))
        
    # Convenience methods for common broadcasts
    
    async def broadcast_speaker_status(self, current_speaker=None, is_speaking=None, 
                                      is_processing=None, pending_speaker=None, 
                                      thinking_sound=None):
        """Broadcast speaker status update."""
        # Update internal state
        if current_speaker is not None:
            self.current_state["current_speaker"] = current_speaker
        if is_speaking is not None:
            self.current_state["is_speaking"] = is_speaking
        if is_processing is not None:
            self.current_state["is_processing"] = is_processing
        if pending_speaker is not None:
            self.current_state["pending_speaker"] = pending_speaker
        if thinking_sound is not None:
            self.current_state["thinking_sound"] = thinking_sound
            
        await self.broadcast(UIMessage(
            type="speaker_status",
            data=self.current_state
        ))
        
    async def broadcast_transcription(self, speaker: str, text: str, 
                                    is_final: bool = False, is_interim: bool = False):
        """Broadcast transcription update."""
        await self.broadcast(UIMessage(
            type="transcription",
            data={
                "speaker": speaker,
                "text": text,
                "is_final": is_final,
                "is_interim": is_interim,
                "timestamp": time.time()
            }
        ))
        
    async def broadcast_ai_stream(self, speaker: str, text: str, 
                                 is_complete: bool = False, session_id: str = ""):
        """Broadcast AI response stream."""
        await self.broadcast(UIMessage(
            type="ai_stream",
            data={
                "speaker": speaker,
                "text": text,
                "is_complete": is_complete,
                "session_id": session_id,
                "timestamp": time.time()
            }
        ))
        
    async def broadcast_system_status(self):
        """Broadcast system status update."""
        await self.broadcast(UIMessage(
            type="system_status",
            data={
                "stt_active": self.current_state["stt_active"],
                "tts_active": bool(self.current_state["is_speaking"]),
                "conversation_active": self.current_state["conversation_active"],
                "current_generation": getattr(self.conversation, '_processing_generation', 0) if self.conversation else 0,
                "director_generation": getattr(self.conversation, '_director_generation', 0) if self.conversation else 0,
                "timestamp": time.time()
            }
        ))
        
    async def broadcast_conversation_update(self, turn_data: Dict[str, Any]):
        """Broadcast conversation history update."""
        await self.broadcast(UIMessage(
            type="conversation_update",
            data={"turn": turn_data}
        ))


# Example usage
if __name__ == "__main__":
    async def test_server():
        server = VoiceUIServer()
        await server.start()
        
        # Keep server running
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            await server.stop()
            
    asyncio.run(test_server())