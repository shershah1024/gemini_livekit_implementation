import asyncio
import base64
import json
import os
from datetime import datetime
from typing import Optional, Any, Dict, List
from livekit import rtc
from zoneinfo import ZoneInfo
import logging
from livekit.agents.log import logger
import websockets

logger = logging.getLogger(__name__)

#Declare the function
async def log_water(amount_ml: int, notes: Optional[str] = None) -> Dict:
    """Log water intake to local SQLite database"""
    try:
        # Get current timestamp in UTC
        timestamp = datetime.now(ZoneInfo("UTC"))
        
        # SQL query to insert water log
        query = """
            INSERT INTO water_logs (timestamp, amount_ml, notes)
            VALUES (?, ?, ?)
        """
        params = (timestamp, amount_ml, notes)

        return {
            "success": True,
            "data": {
                "timestamp": timestamp,
                "amount_ml": amount_ml,
                "notes": notes
            }
        }
    except Exception as e:
        error_msg = f"Error logging water to local db: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": str(e)}




class GeminiRealtimeSession:

    def __init__(
            self,
            api_key: str,
            model: str = "gemini-2.0-flash-exp",
            sample_rate_in: int = 16000,
            sample_rate_out: int = 24000,
            function_call_enabled: bool = True,
            audio_callback=None,
    ):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        self.model = model
        self.sample_rate_in = sample_rate_in
        self.sample_rate_out = sample_rate_out
        self._function_call_enabled = function_call_enabled
        self._audio_callback = audio_callback
        self._first_audio = True
        timestamp = datetime.now(ZoneInfo("UTC"))
        base_instructions = f"""You are an assistant focused on helping your user log water intake. Do not respond to this message. Just start with hello and a greeting. The time is {timestamp}

"""
        
        self.instructions = base_instructions
    
        # Gemini websocket endpoint
        self.host = "generativelanguage.googleapis.com"
        self.uri = f"wss://{self.host}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"

        self._session = None  # Will hold websocket connection
        self._done_event = asyncio.Event()

        self.function_call_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    async def _handle_response(self, response_data: bytes):
        """Handle response from Gemini"""
        try:
            response = json.loads(response_data)

            # Handle server content (audio responses)
            if "serverContent" in response:
                model_turn = response["serverContent"].get("modelTurn", {})
                for part in model_turn.get("parts", []):
                    if "inlineData" in part:
                        inline_data = part["inlineData"]
                        mime_type = inline_data.get("mimeType", "")
                        if mime_type.startswith("audio/"):
                            audio_data = base64.b64decode(inline_data["data"])
                            if self._audio_callback:
                                await self._audio_callback(audio_data, mime_type)
                            else:
                                logger.warning("Received audio data but no callback set")
                    elif "functionCall" in part:
                        await self._handle_tool_call({
                            'functionCalls': [part["functionCall"]]
                        })

            # Handle tool calls
            if 'toolCall' in response:
                await self._handle_tool_call(response['toolCall'])

        except Exception as e:
            logger.error(f"Error handling Gemini response: {e}", exc_info=True)

    async def _handle_tool_call(self, tool_call: Dict):
        """Handle tool calls from Gemini"""
        try:
            responses = []
            for fc in tool_call.get('functionCalls', []):
                function_name = fc['name']
                args = fc.get('args', {})

                # Map function names to actual functions
                function_map = {
                    
                    'log_water': log_water,
                    
                }

                if function_name in function_map:
                    try:
                        result = await function_map[function_name](**args)
                    except Exception as e:
                        error_msg = f"Error executing function {function_name}: {e}"
                        logger.error(error_msg)
                        result = {"success": False, "error": str(e)}
                else:
                    result = {"success": True}

                response = {
                    'id': fc.get('id', ''),
                    'name': function_name,
                    'response': {'result': result}
                }
                responses.append(response)

            # Send tool response back
            msg = {
                'tool_response': {
                    'function_responses': responses
                }
            }

            await self._session.send(json.dumps(msg))

        except Exception as e:
            logger.error(f"Error in response handler: {e}")
            await self._session.send(json.dumps({
                'tool_response': {
                    'function_responses': [{
                        'response': {'result': {"success": False, "error": str(e)}}
                    }]
                }
            }))

    async def send_audio(self, audio_data: bytes | rtc.AudioFrame):
        """Send audio data to Gemini"""
        if not self._session:
            logger.error("No active session")
            return

        try:
            # Convert AudioFrame to bytes if needed
            if isinstance(audio_data, rtc.AudioFrame):
                audio_bytes = audio_data.data.tobytes()
            else:
                audio_bytes = audio_data

            # Send the audio data
            msg = {
                "realtime_input": {
                    "media_chunks": [{
                        "data": base64.b64encode(audio_bytes).decode(),
                        "mime_type": "audio/pcm"
                    }]
                }
            }
            await self._session.send(json.dumps(msg))
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}", exc_info=True)

    async def close(self):
        """Close the session"""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("Closed Gemini session")

    async def send_system_message(self, instructions: str):
        """Send a system message to Gemini"""
        try:
            msg = {
                "client_content": {
                    "turn_complete": True,
                    "turns": [{"role": "system", "parts": [{"text": instructions}]}],
                }
            }
            await self._session.send(json.dumps(msg))
            logger.debug("Sent system message with instructions")
        except Exception as e:
            logger.error(f"Error sending system message: {e}", exc_info=True)

    async def connect(self):
        """Connect to Gemini websocket and send setup message"""
        try:
            self._session = await websockets.connect(self.uri)
            logger.info("Connected to Gemini websocket")

            # Send setup message with tools configuration
            setup_msg = {
                'setup': {
                    'model': f"models/{self.model}",
                    'tools': [
                        {
                            'function_declarations': [
                                
                                {
                                    'name': 'log_water',
                                    'description': 'Log water intake to local SQLite database',
                                    'parameters': {
                                        'type': 'object',
                                        'properties': {
                                            'amount_ml': {'type': 'integer', 'description': 'Amount of water in milliliters'},
                                            'notes': {'type': 'string', 'description': 'Optional notes about water intake'}
                                        },
                                        'required': ['amount_ml']
                                    }
                                },
                                
                            ]
                        }
                    ] if self._function_call_enabled else []
                }
            }

            # Send setup message and wait for response
            await self._session.send(json.dumps(setup_msg))
            setup_response = await self._session.recv()
            setup_data = json.loads(setup_response)
            logger.debug(f"Setup response: {setup_data}")

            # Only proceed if setup was successful
            if setup_data:
                # Start response handler
                self._response_task = asyncio.create_task(self._response_handler())

                # Send initial system message with instructions
                initial_msg = {
                    "client_content": {
                        "turn_complete": True,
                        "turns": [
                            {"role": "system", "parts": [{"text": self.instructions}]}
                        ]
                    }
                }

                await self._session.send(json.dumps(initial_msg))
                logger.info("Setup complete, ready for audio input")
            else:
                logger.error("Setup failed: No response data received")
                raise Exception("Setup failed: No response data received")

        except Exception as e:
            logger.error(f"Failed to connect to Gemini: {e}")
            raise

    async def send_text(self):
        try:
            while True:
                text = await asyncio.to_thread(input, "message > ")
                if text.lower() == "q":
                    break

                msg = {
                    "client_content": {
                        "turn_complete": True,
                        "turns": [
                            {"role": "user", "parts": [{"text": text}]}
                        ]
                    }
                }
                await self._session.send(json.dumps(msg))
                logger.debug("Sent text message")

        except Exception as e:
            logger.error(f"Error in send_text loop: {e}", exc_info=True)

    async def _response_handler(self):
        """Handle incoming messages from Gemini"""
        try:
            while True:
                if not self._session:
                    break
                response = await self._session.recv()
                await self._handle_response(response)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Gemini websocket connection closed")
        except Exception as e:
            logger.error(f"Error in response handler: {e}", exc_info=True)

    async def _send_function_responses(self, responses: List[Dict]):
        """Send function responses back to Gemini"""
        try:
            msg = {
                'functionResponse': responses[0] if len(responses) == 1 else responses
            }

            logger.debug(f"Sending function response: {json.dumps(msg)}")
            await self._session.send(json.dumps(msg))

        except Exception as e:
            logger.error(f"Error sending function responses: {e}", exc_info=True)

    async def send_function_response(self, function_id: str, name: str, result: Any):
        """Send a function response back to Gemini"""
        try:
            msg = {
                'tool_response': {
                    'function_responses': [{
                        'id': function_id,
                        'name': name,
                        'response': {'result': result}
                    }]
                }
            }
            await self._session.send(json.dumps(msg))
            logger.debug(f"Sent function response for {name}")
        except Exception as e:
            logger.error(f"Error sending function response: {e}", exc_info=True)

