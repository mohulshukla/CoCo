import os
import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv
from io import BytesIO
import base64
from elevenlabs.client import ElevenLabs
import moviepy as mp
import json
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

class StoryVideoGenerator:
    def __init__(self, enhanced_dir="enhanced_drawings", output_dir="story_videos"):
        self.enhanced_dir = enhanced_dir
        self.output_dir = output_dir
        self.temp_dir = "temp_processing"
        
        # Create directories
        for directory in [self.output_dir, self.temp_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Video settings
        self.fps = 30
        self.resolution = (1280, 720)  # HD resolution
        self.transition_duration = 2  # seconds
        self.scene_duration = 5  # seconds per scene
        
        # Initialize Gemini
        if GEMINI_API_KEY:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Initialize ElevenLabs
        if ELEVENLABS_API_KEY:
            self.eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        else:
            raise ValueError("ELEVENLABS_API_KEY not set in environment variables")
    
    def get_recent_images(self, limit=5):
        """Get the most recent images from the enhanced drawings directory."""
        if not os.path.exists(self.enhanced_dir):
            raise FileNotFoundError(f"Enhanced images directory '{self.enhanced_dir}' not found!")
        
        image_files = [f for f in os.listdir(self.enhanced_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Sort by creation date (newest first)
        image_files.sort(key=lambda x: os.path.getctime(os.path.join(self.enhanced_dir, x)), 
                        reverse=True)
        
        # Limit to 5 images
        image_files = image_files[:limit]
        return [os.path.join(self.enhanced_dir, f) for f in image_files]
    
    def analyze_images(self, image_paths):
        """Use Gemini to analyze images and create a coherent story."""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in environment variables")
        
        logging.info("Starting image analysis with Gemini...")
        
        # First, get detailed descriptions of each image
        image_descriptions = []
        for i, path in enumerate(image_paths):
            logging.info(f"Analyzing image {i+1}/{len(image_paths)}...")
            try:
                # Load image using PIL
                img = Image.open(path)
                
                # Create the prompt for image analysis
                prompt = """Describe this image in a fun and colorful way. Include:
                - What's happening in the scene
                - The main colors and shapes
                - Any interesting details or characters
                Keep it to 2-3 sentences and use simple, clear language."""
                
                # Generate content using the correct model and API structure
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[prompt, img]
                )
                
                description = response.text.strip()
                logging.info(f"Description for image {i+1}: {description}")
                image_descriptions.append(description)
            except Exception as e:
                logging.error(f"Error analyzing image {path}: {e}")
                image_descriptions.append("A mysterious scene unfolds.")
        
        # Now, generate a coherent story connecting all images
        logging.info("Generating coherent story...")
        story_prompt = f"""
        Create a fun and engaging story that connects these {len(image_descriptions)} scenes. Make it colorful and exciting!
        
        Scene descriptions:
        {chr(10).join(f"Scene {i+1}: {desc}" for i, desc in enumerate(image_descriptions))}
        
        Your response must be a valid JSON object with exactly this structure:
        {{
            "title": "A fun and catchy title",
            "story": "A 4-5 sentence story that connects the scenes. Make it exciting and colorful!",
            "scene_descriptions": [
                "A fun and colorful description of each scene",
                ...
            ]
        }}
        
        Make the story exciting and full of color! Use simple words but make it fun and engaging.
        IMPORTANT: Your response must be valid JSON only, with no additional text or markdown formatting.
        """
        
        try:
            # Generate story using text-only model
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[story_prompt]
            )
            
            # Clean the response text
            response_text = response.text.strip()
            
            # Remove any markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Remove any leading/trailing whitespace or characters before the opening brace
            response_text = response_text[response_text.find("{"):response_text.rfind("}")+1]
            
            logging.info("Cleaned response: %s", response_text)
            
            # Parse the response
            story_data = json.loads(response_text)
            logging.info("Generated story: %s", story_data)
            return story_data
        except Exception as e:
            logging.error(f"Error generating story: {e}")
            logging.error("Response text: %s", response.text)
            return {
                "title": "A Story of Imagination",
                "story": "A tale unfolds through these magical scenes.",
                "scene_descriptions": image_descriptions
            }
    
    def create_audio(self, text, output_path):
        """Generate audio narration using ElevenLabs."""
        try:
            logging.info("Starting audio generation...")
            start_time = time.time()
            
            # Use ElevenLabs for high-quality voice generation
            audio_generator = self.eleven_client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",  # You can change this to any voice ID you prefer
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )
            
            # Convert generator to bytes
            audio_bytes = b''.join(audio_generator)
            
            # Save the audio file
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            
            duration = time.time() - start_time
            logging.info(f"Audio generation completed in {duration:.2f} seconds")
            return True
        except Exception as e:
            logging.error(f"Error creating audio: {e}")
            return False
    
    def apply_ken_burns_effect(self, frame, progress, direction='zoom_out'):
        """Apply Ken Burns effect to a frame with smoother movement."""
        h, w = frame.shape[:2]
        
        # Smoother zoom using easing function
        if direction == 'zoom_out':
            scale = 1.0 + (progress * progress * 0.2)  # Quadratic easing for smoother zoom
        else:
            scale = 1.2 - (progress * progress * 0.2)  # Quadratic easing for smoother zoom
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image with better interpolation
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate crop region
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        x2 = x1 + w
        y2 = y1 + h
        
        # Crop to original size
        return resized[y1:y2, x1:x2]
    
    def apply_pan_effect(self, frame, progress, direction='right'):
        """Apply panning effect to a frame with smoother movement."""
        h, w = frame.shape[:2]
        
        # Smoother pan using easing function
        if direction == 'right':
            offset = int(w * (progress * progress))  # Quadratic easing
            return frame[:, offset:offset+w]
        else:
            offset = int(w * (1 - (progress * progress)))  # Quadratic easing
            return frame[:, offset:offset+w]
    
    def create_scene_clip(self, image_path, duration, effect_type='ken_burns'):
        """Create a video clip for a single scene with effects."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize image to video resolution
        img = cv2.resize(img, self.resolution)
        
        # Create frames for the scene
        frames = []
        for i in range(int(duration * self.fps)):
            progress = i / (duration * self.fps)
            
            if effect_type == 'ken_burns':
                frame = self.apply_ken_burns_effect(img, progress)
            elif effect_type == 'pan':
                frame = self.apply_pan_effect(img, progress)
            else:
                frame = img.copy()
            
            # Ensure frame is the correct size
            frame = cv2.resize(frame, self.resolution)
            frames.append(frame)
        
        return frames
    
    def create_transition(self, frame1, frame2, progress):
        """Create a smooth transition between two frames with easing."""
        # Ensure both frames have the same size and number of channels
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # Use quadratic easing for smoother fade
        eased_progress = progress * progress
        return cv2.addWeighted(frame1, 1 - eased_progress, frame2, eased_progress, 0)
    
    def generate_video(self):
        """Generate the complete story video."""
        try:
            # Get recent images
            image_paths = self.get_recent_images()
            if not image_paths:
                raise ValueError("No images found in enhanced_drawings directory")
            
            logging.info(f"Found {len(image_paths)} images")
            
            # Generate story
            story_data = self.analyze_images(image_paths)
            
            # Create timestamp for unique filenames
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Generate single audio file for the entire narration
            logging.info("Generating audio narration...")
            
            # Combine text into narration (without scene descriptions)
            full_narration = f"{story_data['title']}. {story_data['story']}"
            
            # Generate audio file
            audio_path = os.path.join(self.temp_dir, f"narration_{timestamp}.mp3")
            self.create_audio(full_narration, audio_path)
            
            # Create video frames
            logging.info("Creating video frames...")
            all_frames = []
            
            # Add title screen with fade in
            title_frames = self.create_scene_clip(image_paths[0], 3, 'ken_burns')
            title_text = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            cv2.putText(title_text, story_data["title"], 
                       (self.resolution[0]//4, self.resolution[1]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            all_frames.extend(title_frames)
            
            # Add scenes with transitions
            for i, img_path in enumerate(image_paths):
                logging.info(f"Processing scene {i+1}/{len(image_paths)}")
                # Create scene frames with alternating effects
                scene_frames = self.create_scene_clip(img_path, self.scene_duration, 
                                                    'ken_burns' if i % 2 == 0 else 'pan')
                
                # Add transition if not the first scene
                if i > 0:
                    transition_frames = []
                    for j in range(int(self.transition_duration * self.fps)):
                        progress = j / (self.transition_duration * self.fps)
                        transition_frame = self.create_transition(
                            all_frames[-1], scene_frames[0], progress)
                        transition_frames.append(transition_frame)
                    all_frames.extend(transition_frames)
                
                all_frames.extend(scene_frames)
            
            # Create video clip
            logging.info("Creating final video...")
            video_clip = mp.ImageSequenceClip(all_frames, fps=self.fps)
            
            # Load audio clip
            audio_clip = mp.AudioFileClip(audio_path)
            
            # Set audio to video
            final_video = video_clip.with_audio(audio_clip)
            
            # Write output file with better quality settings
            output_path = os.path.join(self.output_dir, f"story_{timestamp}.mp4")
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=os.path.join(self.temp_dir, 'temp-audio.m4a'),
                remove_temp=True,
                threads=4,
                preset='medium',
                fps=self.fps,
                bitrate='5000k'  # Higher bitrate for better quality
            )
            
            # Clean up
            video_clip.close()
            audio_clip.close()
            
            logging.info(f"Video saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error generating video: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    generator = StoryVideoGenerator()
    video_path = generator.generate_video()
    
    if video_path:
        print(f"Successfully created video: {video_path}")
        # Try to open the video
        try:
            import platform
            import subprocess
            
            system = platform.system()
            if system == 'Darwin':  # macOS
                subprocess.call(('open', video_path))
            elif system == 'Windows':
                os.startfile(video_path)
            else:  # Linux
                subprocess.call(('xdg-open', video_path))
        except:
            pass
    else:
        print("Failed to create video")

if __name__ == "__main__":
    main() 