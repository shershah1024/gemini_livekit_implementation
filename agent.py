from __future__ import annotations
import asyncio
import logging
from dotenv import load_dotenv
import os


from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
    utils,
)
from gemini_realtime_agent import MultimodalAgent

load_dotenv()
logger = logging.getLogger("gemini_agent")
logger.setLevel(logging.ERROR)


def run_multimodal_agent(
        ctx: JobContext,
        participant: rtc.RemoteParticipant,
) -> MultimodalAgent:
    input_audio_ch = utils.aio.Chan[rtc.AudioFrame]()
    agent = MultimodalAgent(
        room=ctx.room,
        input_audio_ch=input_audio_ch,
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
    )
    agent.start(ctx.room, participant)
    return agent


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    participant = await ctx.wait_for_participant()
    logger.info("Participant joined, starting agent")

    agent = run_multimodal_agent(ctx, participant)

    try:
        session_check_counter = 0
        while True:
            # Periodic session check and reconnect if needed
            session_check_counter += 1
            if session_check_counter >= 10:  # Check every 10 seconds
                session_check_counter = 0
                if not agent._session._session:
                    logger.warning("Agent session disconnected, attempting reconnect")
                    try:
                        await agent._session.connect()
                        await agent._session.start_audio_out()
                        logger.info("Successfully reconnected agent session")
                        continue
                    except Exception as e:
                        logger.error(f"Failed to reconnect agent session: {e}")
                        break

            # Check room connection
            if not ctx.room.isconnected():
                logger.error("Room connection lost")
                break

            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error in call loop: {e}")
    finally:
        logger.info("Session finished, shutting down")
        try:
            await agent.stop()
        except Exception as e:
            logger.error(f"Error during agent cleanup: {e}")
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM)) 