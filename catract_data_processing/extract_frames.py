import cv2
import os
import glob


def extract_frames(video_path, output_folder, frame_rate=5):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video
    video_capture = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Calculate the interval between frames (in frames)
    frame_interval = int(fps * frame_rate)

    # Initialize frame counter and index for saving images
    frame_count = 0
    idx = 1

    # Remove the underscores from the video name
    video_basename = os.path.basename(video_path).split('.')[0].replace('_', '')

    while True:
        # Read the next frame
        success, frame = video_capture.read()

        # Break the loop if there are no more frames
        if not success:
            break

        # Save the frame if it is at the correct interval
        if frame_count % frame_interval == 0:
            # Create the filename
            filename = f"{video_basename}_{idx:02d}.png"
            filepath = os.path.join(output_folder, filename)

            # Save the frame as a PNG file
            cv2.imwrite(filepath, frame)

            print(f"Saved: {filepath}")

            # Increment the index
            idx += 1

        # Increment the frame counter
        frame_count += 1

    # Release the video capture object
    video_capture.release()


def process_videos_in_folder(folder_path):
    # Search for all video files in the specified folder
    video_files = glob.glob(
        os.path.join(folder_path, "*.mp4"))  # You can add other extensions if needed (e.g., "*.avi")

    for video_filename in video_files:
        # Set the output directory based on the video filename
        output_dir = os.path.join(os.path.splitext(video_filename)[0], "img")

        # Extract frames for the current video
        extract_frames(video_filename, output_dir)


# Example usage: Provide the path to the folder containing your videos
folder_path = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/video_catarct1k"  # Update this with the actual path to your folder
# folder_path = "/Users/mahtab/Downloads/SSL4MIS/segmentation_env/catract_data_processing/video_data"  # Update this with the actual path to your folder
process_videos_in_folder(folder_path)
