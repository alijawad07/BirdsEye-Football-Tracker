import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from elements.perspective_transform import Perspective_Transform
from arguments import Arguments
import os
import cv2
import numpy as np


def transform_matrix(matrix, p, vid_shape, gt_shape):
    p = (p[0] * 1280 / vid_shape[1], p[1] * 720 / vid_shape[0])
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
            (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
            (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))

    p_after = (int(px * gt_shape[1] / 115), int(py * gt_shape[0] / 74))

    return p_after

def detect_color(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges for each team (adjust the ranges as needed)
    team1_lower = np.array([0, 50, 50])  # Lower range for team 1 color (e.g., red)
    team1_upper = np.array([10, 255, 255])  # Upper range for team 1 color

    team2_lower = np.array([100, 50, 50])  # Lower range for team 2 color (e.g., blue)
    team2_upper = np.array([130, 255, 255])  # Upper range for team 2 color

    # Create masks for each team color
    team1_mask = cv2.inRange(hsv_img, team1_lower, team1_upper)
    team2_mask = cv2.inRange(hsv_img, team2_lower, team2_upper)

    # Count the number of pixels for each team color
    team1_count = cv2.countNonZero(team1_mask)
    team2_count = cv2.countNonZero(team2_mask)

    # Assign the color based on the team with the higher pixel count
    if team1_count > team2_count:
        assigned_color = (0, 0, 255)  # Team 1 color (e.g., red)
    else:
        assigned_color = (255, 255, 255)  # Team 2 color (e.g., white)

    return assigned_color

# Interpolate between two positions
def interpolate_positions(pos1, pos2, factor=0.5):
    x1, y1 = pos1
    x2, y2 = pos2
    new_x = int(x1 + (x2 - x1) * factor)
    new_y = int(y1 + (y2 - y1) * factor)
    return (new_x, new_y)

def main(opt):
    CONFIDENCE_THRESHOLD = opt.conf_thresh
    GREEN = (0, 255, 0)
    threshold = 2.1

    # Load models
    detector = YOLO(opt.weights)
    deep_sort = DeepSort(max_age=50)
    perspective_transform = Perspective_Transform()

    # Video capture
    video_cap = cv2.VideoCapture(opt.source)
    frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    frame_width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Create output directory if it doesn't exist
    os.makedirs(opt.output, exist_ok=True)
    output_name = os.path.join(opt.output, "output.mp4")

    # Initialize the video writer object
    writer = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(frame_width), int(frame_height)))

    frame_num = 0

    # Black Image (Soccer Field)
    bg_ratio = int(np.ceil(frame_width / (3 * 115)))
    gt_img = cv2.imread('./inference/black.jpg')
    gt_img = cv2.resize(gt_img, (115 * bg_ratio, 74 * bg_ratio))
    gt_h, gt_w, _ = gt_img.shape

    # Initialize a dictionary to store the last bounding box for each track
    last_info = {}
    #last_circle_positions = {}

    # Initialize a dictionary to store the last circle positions for each track
    last_circle_positions = {}
    
    while True:
        start = datetime.datetime.now()
        ret, frame = video_cap.read()

        if not ret:
            break

        bg_img = gt_img.copy()
        main_frame = frame.copy()
        yoloOutput = detector(frame)[0]
        # initialize the list of bounding boxes and confidences
        results = []

        # Output: Homography Matrix and Warped image
        if frame_num % 2 == 0:  # Calculate the homography matrix every 5 frames
            M, warped_image = perspective_transform.homography_matrix(main_frame)

        if yoloOutput:

            for data in yoloOutput.boxes.data.tolist():
                # extract the confidence (i.e., probability) associated with the prediction
                confidence = data[4]
                class_id = int(data[5])

                # if the confidence is greater than the minimum confidence,
                # get the bounding box and the class id
                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                class_id = int(data[5])

                x_center = (xmin + xmax) / 2
                y_center = ymax

                if class_id == 1:  # Assuming class_id 1 corresponds to 'ball'
                    coords = transform_matrix(M, (x_center, y_center), (frame_height, frame_width), (gt_h, gt_w))
                    cv2.circle(bg_img, coords, 8, (255, 0, 0), -1)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

                # filter out weak detections by ensuring the 
                # confidence is greater than the minimum confidence
                if (float(confidence) < CONFIDENCE_THRESHOLD):
                    continue
                
                
                    

                # add the bounding box (x, y, w, h), confidence, class id, and center point to the results list
                if class_id == 0:
                    results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id, [(xmin + xmax) // 2, (ymin + ymax) // 2]])


            

            # Tracking
            tracks = deep_sort.update_tracks(results, frame=frame)

            # Loop over the tracks
            for track in tracks:
                # If the track is not confirmed, ignore it
                if not track.is_confirmed():
                    continue

                # Get the track ID, bounding box, and class ID
                track_id = track.track_id
                ltrb = track.to_ltrb()
                xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

                center = (int((xmin + xmax) // 2), int((ymin + ymax) // 2))

                # Perform color detection for the track
                if class_id == 0:  # Assuming class_id 0 corresponds to 'player'
                    coords = transform_matrix(M, center, (frame_height, frame_width), (gt_h, gt_w))
                    try:
                        color = detect_color(main_frame[ymin:ymax, xmin:xmax])
                        cv2.circle(bg_img, coords, bg_ratio + 5, color, -1)
                        color_tuple = tuple(map(int, color))
                        
                        # Draw the track ID on the detection box
                        label = f"Player: {track_id}"
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color_tuple, 2)
                        cv2.rectangle(frame, (xmin, ymin - 25), (xmin + len(label) * 7, ymin - 5), color_tuple, -1)
                        cv2.putText(frame, label, (xmin + 5, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)


                        # Store the circle position in the last_circle_positions dictionary
                        if track_id in last_circle_positions:
                            # Use interpolation to smooth the circle movement
                            last_pos = last_circle_positions[track_id]
                            new_pos = interpolate_positions(last_pos, coords)
                            last_circle_positions[track_id] = new_pos
                            coords = new_pos
                            # Store the bounding box coordinates and color in the last_info dictionary
                            last_info[track_id] = {"bbox": (xmin, ymin, xmax, ymax), "color": color, "coords" : coords}
                        else:
                            last_circle_positions[track_id] = coords

                        if color_tuple == (0, 0, 255):
                            # Draw the circle with track ID
                            circle_radius = 12
                            cv2.circle(bg_img, coords, circle_radius, color_tuple, -1)
                            cv2.putText(bg_img, str(track_id), (coords[0] - 8, coords[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if color_tuple == (255, 255, 255):
                            # Draw the circle with track ID
                            circle_radius = 12
                            cv2.circle(bg_img, coords, circle_radius, color_tuple, -1)
                            cv2.putText(bg_img, str(track_id), (coords[0] - 8, coords[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    except:
                        pass
            
            # Draw the last bounding box and track ID for each track from the previous frame
            for track_id, info in last_info.items():
                xmin, ymin, xmax, ymax = info["bbox"]
                color = info["color"]
                coords = info["coords"]
                color_tuple = tuple(map(int, color))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color_tuple, 2)
                
                label = f"Player: {track_id}"
                cv2.rectangle(frame, (xmin, ymin - 25), (xmin + len(label) * 7, ymin - 5), color_tuple, -1)
                cv2.putText(frame, label, (xmin + 5, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                if color_tuple == (0, 0, 255):
                    # Draw the circle with track ID in green
                    circle_radius = 12
                    #coords = transform_matrix(M, ((xmin + xmax) // 2, (ymin + ymax) // 2), (frame_height, frame_width), (gt_h, gt_w))
                    cv2.circle(bg_img, coords, circle_radius, color_tuple, -1)
                    cv2.putText(bg_img, str(track_id), (coords[0] - 8, coords[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif color_tuple == (255, 255, 255):
                    # Draw the circle with track ID in black
                    circle_radius = 12
                    #coords = transform_matrix(M, ((xmin + xmax) // 2, (ymin + ymax) // 2), (frame_height, frame_width), (gt_h, gt_w))
                    cv2.circle(bg_img, coords, circle_radius, color_tuple, -1)
                    cv2.putText(bg_img, str(track_id), (coords[0] - 8, coords[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


        # Calculate the region of interest (ROI) for placing the bg_img
        roi = frame[-bg_img.shape[0]:, -bg_img.shape[1]:]

        # Make bg_img transparent
        alpha = 0.7  # Transparency level (adjust as needed)
        blended_roi = cv2.addWeighted(roi, 1 - alpha, bg_img, alpha, 0)

        # Replace the ROI in the frame with the blended image
        frame[-bg_img.shape[0]:, -bg_img.shape[1]:] = blended_roi

        model_label = "Model: Yolov8x"
        tracker_label =  "Tracker: DeepSort"
        cv2.rectangle(frame, (10, int(frame_height) - 80), (300, int(frame_height) - 10), (0, 0, 0))
        cv2.putText(frame, model_label, (10, int(frame_height) - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)
        cv2.putText(frame, tracker_label, (10, int(frame_height) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)

        writer.write(frame)
        frame_num += 1

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer
    video_cap.release()
    writer.release()

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = Arguments().parse()
    main(opt)
