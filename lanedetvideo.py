import cv2 as cv
import numpy as np
import math

def region_of_interest(frame, vertices):
    mask = np.zeros_like(frame)
    match_mask_color = 255  # Single channel mask for grayscale
    cv.fillPoly(mask, vertices, match_mask_color)
    return cv.bitwise_and(frame, mask)

def draw_lines(img, lines, color=(0, 0, 255), thickness=5):
    if lines is None:
        return img  # Return the original image if no lines are detected

    line_image = np.zeros_like(img)
    left_fit = []
    right_fit = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            if math.fabs(slope) < 0.5:  # Ignore shallow slopes
                continue
            if slope < 0:  # Left line
                left_fit.append((x1, y1, x2, y2))
            else:  # Right line
                right_fit.append((x1, y1, x2, y2))

    def average_lines(lines):
        if not lines:
            return None
        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        poly = np.polyfit(y_coords, x_coords, deg=1)
        return poly

    left_fit_poly = average_lines(left_fit)
    right_fit_poly = average_lines(right_fit)

    y1 = img.shape[0]  # Bottom of the frame
    y2 = int(y1 * 0.6)  # Slightly below the middle

    def create_coordinates(poly, y1, y2):
        if poly is None:
            return None
        x1 = int(np.polyval(poly, y1))
        x2 = int(np.polyval(poly, y2))
        return (x1, y1, x2, y2)

    left_line = create_coordinates(left_fit_poly, y1, y2)
    right_line = create_coordinates(right_fit_poly, y1, y2)

    if left_line:
        cv.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), color, thickness)
    if right_line:
        cv.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), color, thickness)

    return cv.addWeighted(img, 0.8, line_image, 1.0, 0.0)

def process_frame(frame):
    if frame is None:
        raise ValueError("Received an empty frame!")

    height, width = frame.shape[:2]
    region_of_interest_vertices = [
        (0, height),
        (width // 2, height // 2),
        (width, height),
    ]

    # Preprocessing
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur_frame = cv.GaussianBlur(gray_frame, (5, 5), 0)  # Reduce noise
    canny_frame = cv.Canny(blur_frame, 40, 150)

    # Masking
    cropped_frame = region_of_interest(
        canny_frame,
        np.array([region_of_interest_vertices], np.int32)
    )

    # Hough Transform
    lines = cv.HoughLinesP(
        cropped_frame,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=20,
        maxLineGap=300
    )

    return draw_lines(frame, lines)

def main():
    input_path = "LANE_DETECTION\challenge.mp4"
    output_path = "output_video.mp4"

    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Video properties
    fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    resolution = (frame_width, frame_height)

    fourcc = cv.VideoWriter_fourcc(*'XVID')  # Codec for output video
    output_video = cv.VideoWriter(output_path, fourcc, fps, resolution)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading a frame.")
            break

        try:
            processed_frame = process_frame(frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        # Write the processed frame and display
        output_video.write(processed_frame)
        cv.imshow("Processed Video", processed_frame)

        if cv.waitKey(1) & 0xFF == '27':  # Press 'Esc' to exit
            break

    cap.release()
    output_video.release()
    cv.destroyAllWindows()
    print(f"Processed video saved as '{output_path}'")

if __name__ == "__main__":
    main()
