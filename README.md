# Bird's Eye Football Tracker: YOLOv8 and DeepSort Integration

Bird's Eye Football Tracker is an advanced computer vision system that integrates YOLOv8 and DeepSort algorithms to provide real-time player tracking and a dynamic bird's eye view of football matches. Enhance player analysis, tactical decision making, and the overall viewing experience with this powerful sports technology solution.

## Features

- Enhanced Player Analysis: By leveraging computer vision techniques, this project enables enhanced player analysis in football games. The system can accurately track and analyze player movements, providing valuable insights into player performance, positioning, and strategy.

- Tactical Decision Making: The bird's eye view and real-time tracking capabilities of this project offer a valuable tool for coaches and analysts. It allows them to observe and assess team formations, player interactions, and game dynamics, aiding in tactical decision making and strategic planning.

- Fan Engagement and Visualization: This project enhances the viewing experience for football fans by providing a dynamic bird's eye view of the game. It offers a unique perspective that allows fans to better understand the flow of the game, player interactions, and exciting moments, thus increasing fan engagement and enjoyment.

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- deep-sort-realtime

## Getting Started

1. Clone the repository:

```
https://github.com/alijawad07/BirdsEye-Football-Tracker
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Download the models from [here](https://docs.google.com/uc?export=download&id=1EaBmCzl4xnuebfoQnxU1xQgNmBy7mWi2) and place them under ```weights/```.

4. Run the bird_eye script:
```
python3 bird_eye.py --source --output --weights --conf-thresh
```
- --source => Path to directory containing video

- --output => Path to save the detection results

- --weights => Path to yolov8 weights file

- --conf-thresh => Confidence Threshold

## Acknowledgments

- Special appreciation to Ultralytics for developing the YOLOv8 model and its integration with the project.

## References

- [YOLOv8](https://github.com/ultralytics/ultralytics)
