from __future__ import annotations

import asyncio
from typing import Optional, Any

from livekit import rtc
from livekit.agents import utils, transcription
from livekit.agents.log import logger
from livekit.agents.types import ATTRIBUTE_AGENT_STATE
from livekit.agents.multimodal import agent_playout
from gemini_api import GeminiRealtimeSession


class MultimodalAgent(utils.EventEmitter):

    def __init__(
            self,
            room: rtc.Room,
            input_audio_ch: utils.aio.Chan[rtc.AudioFrame],
            gemini_api_key: str,
            fnc_ctx: Optional[Any] = None,
    ):
        super().__init__()
        self._room = room
        self._input_audio_ch = input_audio_ch
        self._fnc_ctx = fnc_ctx
        self._closed = False
        self._done = asyncio.Event()
        self._started = False
        self._update_state_task = None
        self._read_micro_atask = None
        self._participant = None
        self._subscribed_track = None
        self._playing_handle = None

        # Create Gemini session with callback
        self._session = GeminiRealtimeSession(
            api_key=gemini_api_key,
            audio_callback=self._handle_audio_response  # Pass callback to handle audio directly
        )

    def start(self, room: rtc.Room, participant: rtc.RemoteParticipant | str | None = None) -> None:
        """Start the agent with a room and optional participant"""
        if self._started:
            raise RuntimeError("voice assistant already started")

        room.on("participant_connected", self._on_participant_connected)
        room.on("track_published", self._subscribe_to_microphone)
        room.on("track_subscribed", self._subscribe_to_microphone)

        self._room = room
        self._participant = participant

        if participant is not None:
            if isinstance(participant, rtc.RemoteParticipant):
                self._link_participant(participant.identity)
            else:
                self._link_participant(participant)
        else:
            # no participant provided, try to find the first participant in the room
            for participant in self._room.remote_participants.values():
                self._link_participant(participant.identity)
                break

        # Create a task to wait for initialization and start the main task
        async def _init_and_start():
            try:
                await self._session.connect()
                logger.info("Session initialized")
                self._main_atask = asyncio.create_task(self._main_task())
            except Exception as e:
                logger.exception("Failed to initialize session")
                raise e

        # Schedule the initialization and start task
        asyncio.create_task(_init_and_start())
        self._started = True

    def _subscribe_to_microphone(self, *args, **kwargs) -> None:
        """Subscribe to the participant microphone if found"""
        if self._participant is None:
            return

        participant = self._room.remote_participants.get(
            self._participant if isinstance(self._participant, str) else self._participant.identity)
        if not participant:
            return

        for publication in participant.track_publications.values():
            if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
                continue

            if not publication.subscribed:
                publication.set_subscribed(True)

            if (
                    publication.track is not None
                    and publication.track != self._subscribed_track
            ):
                self._subscribed_track = publication.track
                self._stt_forwarder = transcription.STTSegmentsForwarder(
                    room=self._room,
                    participant=participant,
                    track=self._subscribed_track,
                )

                if self._read_micro_atask is not None:
                    self._read_micro_atask.cancel()

                self._read_micro_atask = asyncio.create_task(
                    self._micro_task(self._subscribed_track)
                )
                break

    def _link_participant(self, participant_identity: str) -> None:
        """Link the agent to a specific participant"""
        participant = self._room.remote_participants.get(participant_identity)
        if participant is None:
            logger.error("_link_participant must be called with a valid identity")
            return

        self._participant = participant
        self._subscribe_to_microphone()

    async def _micro_task(self, track: rtc.LocalAudioTrack) -> None:
        """Process microphone audio from the track"""
        stream_24khz = rtc.AudioStream(track, sample_rate=24000, num_channels=1)
        async for ev in stream_24khz:
            if not self._closed:
                # Convert AudioFrame to bytes before sending
                raw_bytes = ev.frame.data.tobytes()
                self._input_audio_ch.send_nowait(raw_bytes)

    async def _handle_audio_response(self, audio_data: bytes, mime_type: str):
        """Handle audio response from Gemini"""
        if self._closed:
            return

        try:
            # Parse sample rate from mime type
            sample_rate = 24000  # Default to 24kHz
            if "rate=" in mime_type:
                rate_str = mime_type.split("rate=")[1].split(";")[0]
                sample_rate = int(rate_str)

            logger.info(f"Processing audio response: {len(audio_data)} bytes, {mime_type}")

            # Use AudioByteStream to properly handle audio frames
            bstream = utils.audio.AudioByteStream(
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=1200  # Same as in agent_playout.py
            )

            # Process audio through ByteStream
            for frame in bstream.write(audio_data):
                if hasattr(self, '_audio_source'):
                    await self._audio_source.capture_frame(frame)
                    # logger.debug(f"Captured frame with {frame.samples_per_channel} samples at {frame.sample_rate}Hz")
                else:
                    logger.warning("No audio source available to play response")

            # Flush any remaining audio
            for frame in bstream.flush():
                if hasattr(self, '_audio_source'):
                    await self._audio_source.capture_frame(frame)

            # Update state to speaking when we get audio
            self._update_state("speaking")

        except Exception as e:
            logger.error(f"Error handling audio response: {e}", exc_info=True)

    async def _main_task(self):
        # publish audio track for the model's voice
        self._update_state("initializing")
        self._audio_source = rtc.AudioSource(24000, 1)  # 24kHz mono
        logger.info(
            f"Created audio source: {self._audio_source.sample_rate}Hz, {self._audio_source.num_channels} channels")

        self._agent_playout = agent_playout.AgentPlayout(audio_source=self._audio_source)

        # Create and publish the audio track
        track = rtc.LocalAudioTrack.create_audio_track("assistant_voice", self._audio_source)

        # Use minimal publish options
        publish_options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE
        )

        # Publish track and wait for subscription
        self._agent_publication = await self._room.local_participant.publish_track(track, publish_options)
        await self._agent_publication.wait_for_subscription()

        # Meanwhile, read frames from user mic -> pass to gemini
        bstream = utils.audio.AudioByteStream(24000, 1, samples_per_channel=2400)
        async for audio_bytes in self._input_audio_ch:  # Now receiving bytes directly
            if self._closed:
                break
            try:
                # Process the raw bytes through ByteStream
                chunks = list(bstream.write(audio_bytes))

                for chunk in chunks:
                    await self._session.send_audio(chunk)  # Send raw bytes directly

            except Exception as e:
                logger.error(f"Error processing input audio: {str(e)}", exc_info=True)

    def _update_state(self, state: str, delay: float = 0.0):
        """Update the agent's state attribute in the room"""

        @utils.log_exceptions(logger=logger)
        async def _run_task(d: float) -> None:
            await asyncio.sleep(d)
            if self._room and self._room.isconnected():
                await self._room.local_participant.set_attributes({ATTRIBUTE_AGENT_STATE: state})

        if self._update_state_task:
            self._update_state_task.cancel()
        self._update_state_task = asyncio.create_task(_run_task(delay))

    async def stop(self):
        """Graceful shutdown"""
        self._closed = True
        if self._read_micro_atask:
            self._read_micro_atask.cancel()
        if hasattr(self, '_main_atask'):
            self._main_atask.cancel()
        await self._session.close()
        logger.info("MultimodalAgent stopped.")

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        """Handle when a new participant connects to the room"""
        if self._participant is None:
            self._link_participant(participant.identity)
        elif isinstance(self._participant, str) and participant.identity == self._participant:
            self._link_participant(participant.identity)