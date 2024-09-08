import torch
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor

FRAMES_DIR = "/home/jzbumgar/datasets/Depth/Summer2024/20240605/color/"

# This script will run SAM 2 on a video, then convert the outputs to a video

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("x: ", x, ' Y: ', y)

# matplotlib implementation
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Matplot implementation
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def getPrompt(point):
    npPoint = np.array([[point[0], point[1]]], dtype=np.float32)
    # npLabel = np.array([1], np.int32)
    return npPoint

if __name__ == "__main__":
    print("Begin.")

    # Set up necessary torch/cuda settings
    torch.cuda.set_device(2)
    device = torch.device("cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # print("YES TO IF")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = "/home/jzbumgar/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    frame_names = sorted([p for p in os.listdir(FRAMES_DIR)])
    
    # Load a video frame to select the desired chicken
    # frame = cv2.imread(f'{frames_dir}/{frame_names[50]}')
    # cv2.imshow('window_name', frame) 
    # cv2.setMouseCallback('window_name', click_event)
    # cv2.waitKey(0)
    # exit(0)

    inference_state = predictor.init_state(video_path=FRAMES_DIR)

    prompts = {} # Holds the click x/ys for each chicken

    frame_idx = 50 # Frame to interact with

    label = np.array([1], np.int32) # 1 for positive click, 0 for negative click (like MiVOS)

    # Initialize plt figure
    # plt.figure(figsize=(9,6))
    # plt.title(f"frame {frame_idx}")
    # plt.imshow(Image.open(os.path.join(FRAMES_DIR, frame_names[frame_idx])))

    # Add first object to prompts 
    object_id = 1
    points = getPrompt([519, 147])
    prompts[object_id] = points, label

    # Add second object
    object_id = 2
    points = getPrompt([612, 126])
    prompts[object_id] = points, label

    # Add third object
    object_id = 3
    points = getPrompt([1037, 770])
    prompts[object_id] = points, label

    # Run the model
    print(prompts)
    for i in range(len(prompts)):
        obj_id = i+1
        point = prompts[obj_id][0]
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=point,
            labels=label
        )

    for i, out_obj_id in enumerate(out_obj_ids):
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

    # Show the masked frame
    # plt.show()

    # Propage through video, collect results in a dict
    video_segments = {} # Contains per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
        }

    # stride = 30
    plt.close("all")
    # plt.figure(figsize=(6,4))
    
    video_contours = []

    vid_writer = cv2.VideoWriter(
        './20240605.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        20.0,
        (1920, 1080)
    )

    for out_frame_idx, value in tqdm(video_segments.items()):
        # Only show every 30th frame
        # if (out_frame_idx % 30) != 0:
        #     continue 
        # plt.title(f'frame {out_frame_idx}')
        # print(FRAMES_DIR, frame_names[out_frame_idx])
        plt.imshow(Image.open(os.path.join(FRAMES_DIR, frame_names[out_frame_idx])))
        
        # Load the original image
        img_path = FRAMES_DIR + frame_names[out_frame_idx]
        img = cv2.imread(img_path)
        # b_mask = np.zeros(img.shape[:2], np.uint8)

        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)

        # masks = []
        frame_contours = []
        for out_obj_id, out_mask in value.items():
            # print(out_obj_id, type(out_mask))
            # print(out_mask.dtype, out_mask.shape)
            # print(np.unique(out_mask))
            contours, _ = cv2.findContours(out_mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            _ = cv2.drawContours(img, contours, -1, (255, 0, 0), cv2.FILLED)
            #cv2.imshow('contours',img)
            #cv2.waitKey(0)
            #show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            frame_contours.append([out_obj_id, contours])
        video_contours.append(frame_contours)

        # Write frame to video
        vid_writer.write(img)

        #plt.show()

    vid_writer.release()