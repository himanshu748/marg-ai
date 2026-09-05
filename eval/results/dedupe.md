# Detections-to-instance deduplication

Manual ground truth is a visual count of distinct pothole regions across the evidence frames and keyframes, not a frame-by-frame count. Occluded or partially visible regions were counted once.

| Survey | Detections | Instances | Compression | Visible potholes | Unique precision | Unique recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pothole_cars | 21 | 4 | 5.25:1 | 1 | 0.250 | 1.000 |
| pothole_kumasi | 96 | 3 | 32.00:1 | 2 | 0.667 | 1.000 |

| Survey | Per-instance observation counts |
| --- | --- |
| pothole_cars | 2, 1, 4, 14 |
| pothole_kumasi | 62, 32, 2 |
