import cv2
import os

image_folder = '/Users/mohulshukla/Desktop/coco/enhanced_drawings'
output_video = 'output_video.mp4'
fps = 24  # Frames per second
seconds_per_image = 4
frames_per_image = fps * seconds_per_image

# Get list of .png images sorted
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
if not images:
    raise ValueError("No PNG images found in the folder.")

# Get image size
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_frame.shape

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write each image for 4 seconds (repeating frames)
for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)
    if frame.shape != (height, width, 3):
        frame = cv2.resize(frame, (width, height))
    for _ in range(frames_per_image):
        video_writer.write(frame)

video_writer.release()
print(f"✅ Video saved as {output_video}")

