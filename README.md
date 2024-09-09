### SAM 2 UI
A simple UI for SAM 2. Give an input path to a directory of video frames, and the script will let you look through the frames, plot points, then feed the points into SAM 2.

### Arguments:
--input [DIRECTORY]: Directory full of color frames to be processed by SAM 2.

--mode [vid/dir]: Decides if the outputs will be saved as an .mp4, or if they will be saved to a set of directories.

--output [PATH]: Allows user to specify an output filepath. If not specified, the program will default to the current directory. For mode=dir, it should be an existing directory. For mode=vid, should be an mp4 file.

Example command: python3 sam2_ui.py --input [DIRECTORY] --mode vid --output [DIRECTORY]/output.mp4

### Installation:
* Pytorch >= 2.3.1 and Torchvision >= 0.18.1, follow instructions [here.](https://pytorch.org/get-started/locally/)
* OpenCV >= 4.10.0
    * conda install opencv
    * pip install opencv-python
* Follow SAM 2 installation instructions [here](https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#installation).

### UI Controls:
A and D: move forward or back one frame
Left click: Add a point to the frame
Backspace: Remove the most recent point
Esc: Accept current points, proceed to masking and propogation

### Visualizations of UI:
UI itself, three objects clicked on:
![image](https://github.com/user-attachments/assets/3432b3cf-4c0f-4e7d-bb63-456b230a018d)

Mask outputs overlayed onto original frame, taken from the "vid" mode:
![overlay](https://github.com/user-attachments/assets/c2b67cf0-f4b4-48f3-9eeb-04969d191464)

SAM mask displayed as a binary mask, taken from "dir" mode (one image made for each object):
![40_1](https://github.com/user-attachments/assets/7a55bf01-73ab-47d7-abb3-850df5de9605)
