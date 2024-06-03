import cv2
import numpy as np

# Calculating the angle between a line segment and the x-axis
def LineAngleWithXAxis(x1, y1, x2, y2):
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angle

# Check if Two Lines are Similar
def is_similar_line(line1, line2, angle_threshold=10, distance_threshold=20):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    angle1 = LineAngleWithXAxis(x1, y1, x2, y2)
    angle2 = LineAngleWithXAxis(x3, y3, x4, y4)
    
    if abs(angle1 - angle2) > angle_threshold:
        return False
    
    distance = np.min([np.hypot(x1 - x3, y1 - y3), np.hypot(x1 - x4, y1 - y4),
                       np.hypot(x2 - x3, y2 - y3), np.hypot(x2 - x4, y2 - y4)])
    
    return distance < distance_threshold

# Merge Two Lines
def merge_lines(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    points = sorted(points)
    
    return [points[0][0], points[0][1], points[-1][0], points[-1][1]]

def line_length(hand):
    """
    Calculate the length of the line segment from coordinates (x1, y1) to (x2, y2).

    Parameters:
    hand (tuple): Coordinates of points (x1, y1, x2, y2).

    Returns:
    float: Line segment length.
    """
    x1, y1, x2, y2 = hand
    length = np.linalg.norm(np.array((x2, y2)) - np.array((x1, y1)))
    return length

def find_farthest_point(center, hand):
    """
    Find the farthest coordinates from the center point in the hand coordinates.

    Parameters:
    center (tuple): Center coordinates (x, y).
    hand (tuple): Coordinates of points (x1, y1, x2, y2).

    Returns:
    tuple: Coordinate furthest from center.
    """
    # Calculate the distance from each (x, y) to the center
    distances = [
        np.linalg.norm(np.array(center) - np.array((hand[0], hand[1]))),
        np.linalg.norm(np.array(center) - np.array((hand[2], hand[3])))
    ]

    # Find the index of the greatest distance
    max_distance_index = np.argmax(distances)

    # Get the coordinates farthest from the center
    farthest_point = (hand[0], hand[1]) if max_distance_index == 0 else (hand[2], hand[3])
    return farthest_point

def calculate_angle(p1, p2):
    """Calculate the angle in degrees between the line p1 to p2 and the horizontal axis."""
    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    return angle if angle >= 0 else angle + 360

def determine_hour_time(center, hour_length, current_hour_point, twelve, three, six, nine):
    """Determine the current time based on the angle of the hour hand."""
    # Calculate angles of hour hand with respect to 12 o'clock
    angle_current_to_twelve = calculate_angle(center, current_hour_point) - calculate_angle(center, twelve)
    angle_current_to_twelve = angle_current_to_twelve if angle_current_to_twelve >= 0 else angle_current_to_twelve + 360
    
    # Convert angle to time
    time = angle_current_to_twelve / 30  # 30 degrees per hour
    time_int = int(time) 
    
    # Format hours with leading zero if less than 10
    formatted_time = f"{time_int:02d}"
    return formatted_time

def determine_min_sec_time(center, hour_length, current_hour_point, twelve, three, six, nine):
    """Determine the current time based on the angle of the hour hand."""
    # Calculate angles of hour hand with respect to 12 o'clock
    angle_current_to_twelve = calculate_angle(center, current_hour_point) - calculate_angle(center, twelve)
    angle_current_to_twelve = angle_current_to_twelve if angle_current_to_twelve >= 0 else angle_current_to_twelve + 360
    
    # Convert angle to time
    time = angle_current_to_twelve / 6  # 6 degrees per min, sec
    time_int = int(time) 
    
    # Format hours with leading zero if less than 10
    formatted_time = f"{time_int:02d}"
    return formatted_time

def locate_time_marks(center, hour_hand_length):
    """Locate the time marks (12, 3, 6, 9) based on the description given."""
    # Approximate positions based on the description
    twelve = (center[0], center[1] - hour_hand_length)
    six = (center[0], center[1] + hour_hand_length)
    three = (center[0] + hour_hand_length, center[1])
    nine = (center[0] - hour_hand_length, center[1])
    
    return twelve, three, six, nine

# Hàm tạo hình chữ nhật bao quanh đoạn thẳng
def create_bounding_box(x1, y1, x2, y2, thickness=5):
    angle = LineAngleWithXAxis(x1, y1, x2, y2)
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    rect = ((x1 + x2) / 2, (y1 + y2) / 2), (length + thickness, thickness), angle
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

# Load image
image_path = 'img/clock10.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (640, 640))
# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
#cv2.imshow("before invert", binary_image)
# Invert image colors
binary_image = cv2.bitwise_not(binary_image)
#cv2.imshow("after invert", binary_image)

# Find contours in the image
contour_info = []
contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    contour_info.append((i, area))

# Sort contours based on the area
contour_info.sort(key=lambda x: x[1], reverse=True)

# Select the third largest contour
third_largest_contour = contours[contour_info[2][0]]
contour_mask = np.zeros_like(binary_image)
cv2.drawContours(contour_mask, [third_largest_contour], -1, (255), thickness=cv2.FILLED)
cv2.imshow('Original', contour_mask)

# Perform thinning using OpenCV's ximgproc module
thinned_image = cv2.ximgproc.thinning(contour_mask)
cv2.imshow('thinned_image', thinned_image)

# Detect lines using Hough transform
lines = cv2.HoughLinesP(thinned_image, 1, np.pi / 180, 15, None, 20, 100)
result_image = image.copy()
line_lengths = []
merged_lines = []
used_lines = set()
for i in range(len(lines)):
    if i in used_lines:
        continue
    x1, y1, x2, y2 = lines[i][0]
    current_line = [x1, y1, x2, y2]
    
    for j in range(i + 1, len(lines)):
        if j in used_lines:
            continue
        x3, y3, x4, y4 = lines[j][0]
        if is_similar_line(current_line, [x3, y3, x4, y4]):
            current_line = merge_lines(current_line, [x3, y3, x4, y4])
            used_lines.add(j)
    
    merged_lines.append(current_line)
    used_lines.add(i)

for line in merged_lines:
    x1, y1, x2, y2 = line
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    line_lengths.append((length, (x1, y1, x2, y2)))

line_lengths.sort(reverse=True, key=lambda x: x[0])
second_hand = line_lengths[0][1]
minute_hand = line_lengths[1][1]
hour_hand = line_lengths[2][1]
print(second_hand)

center = (320,320)
second_coordinate = find_farthest_point(center, second_hand)
minute_coordinate = find_farthest_point(center, minute_hand)
hour_coordinate = find_farthest_point(center, hour_hand)

second_length = line_length(second_hand)
minute_length = line_length(minute_hand)
hour_length = line_length(hour_hand)

# Locate time marks
twelve, three, six, nine = locate_time_marks(center, second_length)
current_time_second = determine_min_sec_time(center, second_length, second_coordinate, twelve, three, six, nine)

# Locate time marks
twelve, three, six, nine = locate_time_marks(center, minute_length)
current_time_minute = determine_min_sec_time(center, minute_length, minute_coordinate, twelve, three, six, nine)

# Locate time marks
twelve, three, six, nine = locate_time_marks(center, hour_length)
current_time_hours = determine_hour_time(center, hour_length, hour_coordinate, twelve, three, six, nine)

text = f"{current_time_hours}:{current_time_minute}:{current_time_second}"

org = (50, 50)  
font = cv2.FONT_HERSHEY_SIMPLEX  
font_scale = 1  # font size
color = (255, 0, 255)  
thickness = 2  
cv2.putText(result_image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
print(hour_hand)
# Take the 3 longest lines
longest_lines = line_lengths[:3]

# blue for seconds, green for minutes, red for hours
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# Draw the lines and their bounding boxes on the result image
for i, (_, line) in enumerate(longest_lines):
    x1, y1, x2, y2 = line
    box = create_bounding_box(x1, y1, x2, y2, thickness=10)
    cv2.drawContours(result_image, [box], 0, colors[i], 2)

cv2.imshow('Result with Lines', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
