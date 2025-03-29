import cv2
import mediapipe as mp
import numpy as np
import os
import time
import base64
from datetime import datetime
from io import BytesIO
import threading
from google import genai
from google.genai import types
from PIL import Image

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize drawing canvas
canvas = None
prev_point = None
drawing_color = (0, 0, 255)  # Red color by default
drawing_thickness = 5
eraser_thickness = 20  # Eraser is thicker than the pen

# Create output directory for saved images
save_dir = "saved_drawings"
os.makedirs(save_dir, exist_ok=True)

# Create directory for Gemini enhanced images
enhanced_dir = "enhanced_drawings"
os.makedirs(enhanced_dir, exist_ok=True)

# Gemini API settings
GEMINI_API_KEY = "AIzaSyAUwKWagFjsciy0etZjxTyUoxk4dighn5M"  # Using the provided API key
GEMINI_MODEL = "gemini-2.0-flash-exp-image-generation"  # Using the specified image generation model

# Initialize enhanced image placeholder
enhanced_image = None
is_processing = False

# Initialize Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Function to enhance drawing with Gemini
def enhance_drawing_with_gemini(drawing, prompt=""):
    global enhanced_image, is_processing
    
    if not GEMINI_API_KEY:
        print("Error: Gemini API key is not set. Cannot enhance drawing.")
        error_img = np.zeros((drawing.shape[0], drawing.shape[1], 3), dtype=np.uint8)
        cv2.putText(error_img, "API KEY NOT SET", (error_img.shape[1]//4, error_img.shape[0]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        enhanced_image = error_img
        is_processing = False
        return
    
    try:
        # Create a thread to handle the Gemini API call
        def process_with_gemini(drawing_img, prompt_text):
            global enhanced_image, is_processing
            try:
                # Default prompt if none provided
                if not prompt_text:
                    prompt_text = "Enhance this sketch into a detailed image."
                
                # Convert OpenCV image to PIL Image
                pil_img = Image.fromarray(cv2.cvtColor(drawing_img, cv2.COLOR_BGR2RGB))
                
                # Convert the image to bytes and then to base64
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Prepare the prompt and image data
                contents = [
                    {"text": prompt_text},
                    {"inlineData": {
                        "mimeType": "image/jpeg",
                        "data": img_str
                    }}
                ]
                
                # Set the model and configuration
                config = types.GenerateContentConfig(response_modalities=['Text', 'Image'])
                
                # Generate the enhanced image
                response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=config
                )
                
                print("RESPONSE: ", response)
                
                # Process the response
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        print("Text response:", part.text)
                    elif part.inline_data is not None:
                        # Save the generated image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        enhanced_path = f"{enhanced_dir}/enhanced_{timestamp}.png"
                        
                        # Convert to PIL image
                        resp_image = Image.open(BytesIO((part.inline_data.data)))
                        resp_image.save(enhanced_path)
                        
                        # Convert PIL Image to OpenCV format
                        img_array = np.array(resp_image)
                        # Convert RGB to BGR (OpenCV format)
                        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        else:
                            img = img_array
                        
                        print(f"Enhanced image saved to {enhanced_path}")
                        
                        # Update the global enhanced image
                        enhanced_image = img
                        return
                
                # If no image was found in the response
                print("No image found in Gemini response")
                error_img = np.zeros((drawing.shape[0], drawing.shape[1], 3), dtype=np.uint8)
                cv2.putText(error_img, "No image in response", (error_img.shape[1]//4, error_img.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                enhanced_image = error_img
            
            except Exception as e:
                print(f"Error enhancing drawing with Gemini: {str(e)}")
                error_img = np.zeros((drawing.shape[0], drawing.shape[1], 3), dtype=np.uint8)
                cv2.putText(error_img, f"ERROR: {str(e)[:30]}...", (10, error_img.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                enhanced_image = error_img
            
            finally:
                is_processing = False
        
        # Start the processing thread
        is_processing = True
        
        # Create a "Processing..." image while waiting
        processing_img = np.zeros((drawing.shape[0], drawing.shape[1], 3), dtype=np.uint8)
        cv2.putText(processing_img, "Processing with Gemini...", (processing_img.shape[1]//4, processing_img.shape[0]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        enhanced_image = processing_img
        
        # Start processing in background
        threading.Thread(target=process_with_gemini, args=(drawing, prompt)).start()
        
    except Exception as e:
        print(f"Error preparing drawing for Gemini: {str(e)}")
        is_processing = False

# Start webcam
cap = cv2.VideoCapture(1)  # Using camera index 1 as specified
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Get initial frame to set canvas size
success, frame = cap.read()
if not success:
    print("Can't receive frame from camera")
    exit()
    
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

# Create a window for the combined view (original + enhanced)
cv2.namedWindow('Hand Drawing', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Drawing', w*2, h)  # Double width to show both images side by side

# Status text variables
mode_text = "Drawing Mode"
color_text = "Red"

while True:
    # Read frame from webcam
    success, frame = cap.read()
    if not success:
        print("Can't receive frame")
        break
        
    # Flip the frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hand landmarks
    results = hands.process(rgb_frame)
    
    # Display canvas on frame
    combined_img = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    
    # Initialize hand mode
    hand_mode = "None"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                combined_img, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            # Get landmarks positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([int(lm.x * w), int(lm.y * h)])
            
            # Calculate finger states - are they extended or not?
            fingers_extended = []
            
            # For thumb (using different method)
            thumb_tip = landmarks[4]
            wrist = landmarks[0]
            index_mcp = landmarks[5]  # Index finger MCP joint
            thumb_extended = thumb_tip[0] < landmarks[3][0] if wrist[0] < index_mcp[0] else thumb_tip[0] > landmarks[3][0]
            fingers_extended.append(thumb_extended)
            
            # For other fingers (compare y-coordinate of tip with PIP joint)
            fingers_extended.append(landmarks[8][1] < landmarks[6][1])   # Index
            fingers_extended.append(landmarks[12][1] < landmarks[10][1]) # Middle
            fingers_extended.append(landmarks[16][1] < landmarks[14][1]) # Ring
            fingers_extended.append(landmarks[20][1] < landmarks[18][1]) # Pinky
            
            # Get current hand position (for drawing or erasing)
            current_point = tuple(landmarks[8])  # Using index finger tip for all modes
            
            # FEATURE 1: DRAWING MODE - Only index finger is extended
            if fingers_extended[1] and not fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
                hand_mode = "Drawing"
                mode_text = "Drawing Mode"
                
                # Draw circle at index finger tip position
                cv2.circle(combined_img, current_point, 10, drawing_color, -1)
                
                # Draw line on canvas
                if prev_point:
                    cv2.line(canvas, prev_point, current_point, drawing_color, drawing_thickness)
                
                prev_point = current_point
                
            # FEATURE 2: ERASER MODE - Closed fist (no fingers extended)
            elif not any(fingers_extended):
                hand_mode = "Erasing"
                mode_text = "Eraser Mode"
                
                # Draw eraser circle
                cv2.circle(combined_img, current_point, eraser_thickness, (255, 255, 255), -1)
                cv2.circle(combined_img, current_point, eraser_thickness, (0, 0, 0), 2)
                
                # Erase from canvas (draw black with thicker line)
                if prev_point:
                    cv2.line(canvas, prev_point, current_point, (0, 0, 0), eraser_thickness)
                
                prev_point = current_point
                
            # FEATURE 3: CLEAR ALL - Middle finger extended only
            elif not fingers_extended[0] and not fingers_extended[1] and fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
                hand_mode = "Clear All"
                mode_text = "Cleared Canvas"
                
                # Clear the entire canvas
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                prev_point = None
                
            else:
                # Any other hand position - stop drawing/erasing
                hand_mode = "None"
                mode_text = "Not Drawing"
                prev_point = None
                
    else:
        # If no hand is detected, reset previous point
        prev_point = None
        hand_mode = "None"
        mode_text = "No Hand Detected"
    
    # Display status information on the frame
    cv2.putText(combined_img, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined_img, f"Color: {color_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, drawing_color, 2)
    
    # Draw guidelines for gestures at the bottom of the screen
    help_y = h - 30
    cv2.putText(combined_img, "Index finger: Draw | Fist: Erase | Middle finger: Clear All", 
                (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(combined_img, "Keyboard: r,g,b,y,p=Colors | G=Gemini enhance | s=Save | q=Quit", 
                (10, help_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Create a combined image to display (original + enhanced)
    display_img = combined_img.copy()
    
    # If we have an enhanced image, add it to the right side
    if enhanced_image is not None:
        # Resize enhanced image to match the canvas height
        enhanced_resized = cv2.resize(enhanced_image, (w, h))
        
        # Create a side-by-side display image
        display_img = np.zeros((h, w*2, 3), dtype=np.uint8)
        display_img[:, :w] = combined_img
        display_img[:, w:] = enhanced_resized
        
        # Add a dividing line
        cv2.line(display_img, (w, 0), (w, h), (255, 255, 255), 2)
        
        # Add an "Enhanced" label on the right side
        cv2.putText(display_img, "Enhanced Drawing", (w + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('Hand Drawing', display_img)
    
    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # Esc or q to quit
        break
    elif key == ord('c'):  # c to clear canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    elif key == ord('s'):  # s to save the current drawing
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/drawing_{timestamp}.png"
        
        # Save the canvas
        cv2.imwrite(filename, canvas)
        
        # Provide feedback on the screen
        print(f"Drawing saved to {filename}")
        
        # Show feedback on screen
        feedback_img = combined_img.copy()
        cv2.putText(feedback_img, "Drawing Saved!", (w//2 - 100, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hand Drawing', feedback_img)
        cv2.waitKey(1000)  # Show message for 1 second
        
        # Clear the canvas for a new drawing
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    elif key == ord('G'):  # capital G to enhance with Gemini
        if not is_processing:
            print("Enhancing drawing with Gemini...")
            enhance_drawing_with_gemini(canvas, "Enhance this sketch into a beautiful, colorful detailed image.")
    elif key == ord('r'):  # r for red
        drawing_color = (0, 0, 255)
        color_text = "Red"
    elif key == ord('g'):  # g for green
        drawing_color = (0, 255, 0)
        color_text = "Green"
    elif key == ord('b'):  # b for blue
        drawing_color = (255, 0, 0)
        color_text = "Blue"
    elif key == ord('y'):  # y for yellow
        drawing_color = (0, 255, 255)
        color_text = "Yellow"
    elif key == ord('p'):  # p for purple
        drawing_color = (255, 0, 255)
        color_text = "Purple"
    elif key == ord('+') or key == ord('='):  # + to increase thickness
        drawing_thickness = min(30, drawing_thickness + 1)
    elif key == ord('-'):  # - to decrease thickness
        drawing_thickness = max(1, drawing_thickness - 1)
    elif key == ord('e'):  # e to toggle eraser size
        eraser_thickness = 40 if eraser_thickness == 20 else 20

# Release resources
cap.release()
cv2.destroyAllWindows()