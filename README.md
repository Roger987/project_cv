# Usage
To run the program, use the following command:

```bash
./main <video_filename> <segmentation_flag> <upvision_flag> <evaluation_flag>
```



# Parameters

> <video_filename>: The path to the input video file.

> <segmentation_flag>: Flag to enable segmentation (1 to enable, 0 to disable).

> <upvision_flag>: Flag to enable upvision (1 to enable, 0 to disable).

> <evaluation_flag>: Flag to enable evaluation (1 to enable, 0 to disable).


*Example*

```bash
./table_tennis_tracking ../Dataset/video.mp4 1 0 1
```


# Output
The processed video will be saved in the ../Dataset/<video_folder>/output.mp4.

*Additional Output*
If evaluation is enabled, the following files will be generated:

> Bounding box coordinates of the first and last frames.

> Masks of the first and last frames.

> Mean IoU and mAP evaluation metrics.
