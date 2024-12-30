# Gemini LiveKit Implementation



## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/shershah1024/gemini_livekit_implementation.git
   cd gemini_livekit_implementation
   ```

2. Set up a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   - Copy `.env.sample` to `.env`:
     ```bash
     cp .env.sample .env
     ```
   - Add your API keys to `.env`:
     - `LIVEKIT_API_KEY`: Your LiveKit API key
     - `LIVEKIT_API_SECRET`: Your LiveKit API secret
     - `LIVEKIT_URL`: Your LiveKit server URL
     - `GEMINI_API_KEY`: Your Google Gemini API key

## Development Notes

1. Gemini function calling works a little differently. You have to define the functions and send them as part of the set up message. I have added a basic example for logging water intake

2. I have not figured out how to change the voice. If anyone can, it would be amazing

3. Planning to add transcription also

4. I have not found a better way to send system messages yet.


Use the livekit sandbox to test
