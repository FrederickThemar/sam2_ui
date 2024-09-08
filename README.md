# sam2_ui
A simple UI for SAM 2. Give an input path to a directory of video frames, and the script will let you look through the frames, plot points, then feed the points into SAM 2.

Arguments:
--input [DIRECTORY]: Directory full of color frames to be processed by SAM 2.
--mode [vid/dir]: Decides if the outputs will be saved as an .mp4, or if they will be saved to a set of directories.
--output [PATH]: Allows user to specify an output filepath. If not specified, the program will default to the current directory. For mode=dir, it should be an existing directory. For mode=vid, should be an mp4 file.

**UI Controls:**
A and D: move forward or back one frame
Left click: Add a point to the frame
Backspace: Remove the most recent point
Esc: Accept current points, proceed to masking and propogation
