import argparse
import torch
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

# Displays all the masked frames one-by-one. 
def display_masks(video_segments, frame_names, frames_dir):
    for out_frame_idx, value in video_segments.items():
        # Load original image
        img = cv2.imread(frames_dir + frame_names[out_frame_idx])

        # Draw the masks to the original image
        for out_obj_id, out_mask in value.items():
            contours, _ = cv2.findContours(out_mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            _ = cv2.drawContours(img, contours, -1, (255, 0, 0), cv2.FILLED)

        # Display the image
        cv2.imshow('masks', img)
        cv2.waitKey(15)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Begin SAM 2 Simple UI.")

    # Set pytorch settings
    torch.cuda.set_device(2)
    device = torch.device("cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Set up the CLI argument intake
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Directory with input frames.")
    parser.add_argument("--save", type=str, help="Mode for saving output. 'vid' for mp4, 'dir' for image directory. If not given, will display output.")
    args = parser.parse_args()
    frames_dir = args.input
    save_mode = args.save

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
    cv2.destroyAllWindows()

    ### Section 2: Run model
    print(f'Chosen frame: {chosenStart}')

    checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    # Load up predictor
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    inference_state = predictor.init_state(video_path=frames_dir)

    # DEBUG: Change later once negative clicks are supported
    label = np.array([1], np.int32)

    for i in range(len(PROMPTS)):
        obj_id = i+1
        # PROMPTS[obj_id] = np.array([[x, y]], dtype=np.float32)
        pre_point = PROMPTS[obj_id]
        point = np.array([[pre_point[0],pre_point[1]]], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=chosenStart,
            obj_id=obj_id,
            points=point,
            labels=label
        )

    # Propogate mask through rest of video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
        }

    ### Section 3: Save model outputs
    if save_mode == "vid":
        print("ERROR: vid_save NOT IMPLEMENTED")
        exit(1)
    elif save_mode == "dir":
        print("ERROR: dir_save NOT IMPLEMENTED")
        exit(1)
    else:
        display_masks(video_segments, frames, frames_dir)