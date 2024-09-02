import argparse
import cv2
import os

import numpy as np

from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor

# Stores the click points for each object
PROMPTS = {}

# When mouse clicked in cv2, add the click point to PROMPTS
def on_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        obj_id = len(PROMPTS) + 1
        # PROMPTS[obj_id] = np.array([[x, y]], dtype=np.float32)
        PROMPTS[obj_id] = (x,y)



if __name__ == '__main__':
    print("Begin SAM 2 Simple UI.")

    # Set up the CLI argument intake
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Directory with input frames.")
    args = parser.parse_args()
    frames_dir = args.input

    # Check if input exists:
    if frames_dir is None:
        print("ERROR: Must include an input directory. Try again.")
        exit(1)
    elif not os.path.isdir(frames_dir):
        print("ERROR: Input directory does not exist. Try again.")
        exit(1)

    ### Section 1: Get the clicks for the model
    # Save the frame names
    frames = sorted([p for p in os.listdir(frames_dir)])
    
    # Display frames until user selects a frame to use for model input
    showFrame = True
    toShow = 0 # Index of frames list to show
    prevShow = 0
    frame = cv2.imread(f'{frames_dir}/{frames[toShow]}')
    orig_frame = frame.copy()
    firstLoop = True

    cv2.imshow('window_name', frame)
    cv2.setMouseCallback('window_name', on_click)
    while showFrame:
        # Load new frame if needed
        if toShow != prevShow:
            frame = cv2.imread(f'{frames_dir}/{frames[toShow]}') 
            orig_frame = frame.copy()
        else:
            frame = orig_frame.copy()      

        # Draw clicked points
        for obj_id, point in PROMPTS.items():
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        # Display the frame
        cv2.imshow('window_name', frame)
        key = cv2.waitKey(1)

        # Respond to user key presses accordingly
        if key == 27: # Escape key, exit loop
            chosenStart = toShow
            showFrame = False
        elif key == 97: # A Key, move back one frame
            if toShow > 0:
                prevShow = toShow
                toShow-=1
                PROMPTS = {} # Must empty points dictionary
        elif key == 100: # D Key, move forward one frame
            if toShow < len(frames)-1:
                prevShow = toShow
                toShow+=1
                PROMPTS = {} # Must empty points dictionary
        elif key == 8 and len(PROMPTS) > 0: # Backspace, remove last point from dict
            popped = PROMPTS.pop(len(PROMPTS))
    
    ### Section 2: Run model
    print("RUN MODEL")

    ### Section 3: Save model outputs