import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def test_audio_generation():
    """Test audio generation with ElevenLabs."""
    try:
        # Initialize ElevenLabs client
        if not ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY not set in environment variables")
        
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        
        # Test text
        test_text = "Hello! This is a test of the ElevenLabs text to speech system. It should sound much better than the previous voices."
        
        logging.info("Generating audio...")
        
        # Generate audio
        audio_generator = client.text_to_speech.convert(
            text=test_text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # Default voice
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        
        # Convert generator to bytes
        audio_bytes = b''.join(audio_generator)
        
        # Save the audio file
        output_path = "test_audio.mp3"
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        
        logging.info(f"Audio saved to: {output_path}")
        
        # Try to play the audio
        logging.info("Playing audio...")
        play(audio_bytes)
        
        return True
        
    except Exception as e:
        logging.error(f"Error in audio generation: {e}")
        return False

if __name__ == "__main__":
    test_audio_generation() 