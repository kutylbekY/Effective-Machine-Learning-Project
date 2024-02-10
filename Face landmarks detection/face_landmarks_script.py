import os
import dlib
import cv2
import math

# Input and output folders
input_folder = 'input'
output_folder = 'output'

# Load the pre-trained shape predictor model
model_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(model_path)

# Load the face detector
detector = dlib.get_frontal_face_detector()

# Process images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG'):
        # Load the image using OpenCV
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray_image)
        if len(faces) > 0:
            face = faces[0]  # Assuming there's only one face in the image
            landmarks = predictor(gray_image, face)
            print(f"Face detected in {filename}")
        else:
            print(f"No face detected in {filename}")
            continue
        
        # Draw landmarks on the original image
        for point in landmarks.parts():
            # Calculate a smaller scaling factor based on the face size
            scaling_factor = face.width() / 500  # Adjust this value as needed
            
            # Calculate the adjusted radius
            adjusted_radius = int(5 * scaling_factor)
            
            cv2.circle(image, (point.x, point.y), adjusted_radius, (0, 255, 0), -1)

        # Calculate the length of the lines based on a fraction of the image width
        line_fraction = 0.3  # Adjust this fraction as needed
        line_length = int(image.shape[1] * line_fraction)

        # Add vertical lines through nose and eyes
        bottom_nose_point = (landmarks.part(33).x, landmarks.part(33).y + line_length)
        top_nose_point = (landmarks.part(27).x, landmarks.part(27).y - line_length)
        
        line_thickness = adjusted_radius  # Set line thickness to be the same as the landmark point size
        
        cv2.line(image, bottom_nose_point, top_nose_point, (255, 0, 0), line_thickness)  # Line from bottom to top of the nose

        # Eyes lines parallel to nose line
        left_eye_middle_x = (landmarks.part(36).x + landmarks.part(39).x) // 2
        left_eye_middle_y = (landmarks.part(36).y + landmarks.part(39).y) // 2

        right_eye_middle_x = (landmarks.part(42).x + landmarks.part(45).x) // 2
        right_eye_middle_y = (landmarks.part(42).y + landmarks.part(45).y) // 2

        nose_line_copy_left_start = (left_eye_middle_x, left_eye_middle_y - line_length)
        nose_line_copy_left_end = (left_eye_middle_x, left_eye_middle_y + line_length)

        nose_line_copy_right_start = (right_eye_middle_x, right_eye_middle_y - line_length)
        nose_line_copy_right_end = (right_eye_middle_x, right_eye_middle_y + line_length)

        # Draw the copied nose line in the middle of both the left and right eyes
        cv2.line(image, nose_line_copy_left_start, nose_line_copy_left_end, (255, 0, 0), line_thickness)
        cv2.line(image, nose_line_copy_right_start, nose_line_copy_right_end, (255, 0, 0), line_thickness)


        # Lines 76 - 108 to make eyes lines based on eyes, not parallel to nose
        # Calculate the middle of the eyes
        # left_eye_middle = ((landmarks.part(36).x + landmarks.part(39).x) // 2,
        #                    (landmarks.part(36).y + landmarks.part(39).y) // 2)
        # right_eye_middle = ((landmarks.part(42).x + landmarks.part(45).x) // 2,
        #                     (landmarks.part(42).y + landmarks.part(45).y) // 2)
        
        # # Calculate the direction vector of the line perpendicular to the eyes
        # left_eye_perpendicular = (landmarks.part(39).y - landmarks.part(36).y, landmarks.part(36).x - landmarks.part(39).x)
        # right_eye_perpendicular = (landmarks.part(45).y - landmarks.part(42).y, landmarks.part(42).x - landmarks.part(45).x)
        
        # # Normalize the direction vectors
        # left_eye_perpendicular_length = math.sqrt(left_eye_perpendicular[0] ** 2 + left_eye_perpendicular[1] ** 2)
        # right_eye_perpendicular_length = math.sqrt(right_eye_perpendicular[0] ** 2 + right_eye_perpendicular[1] ** 2)
        # left_eye_perpendicular = (left_eye_perpendicular[0] / left_eye_perpendicular_length, left_eye_perpendicular[1] / left_eye_perpendicular_length)
        # right_eye_perpendicular = (right_eye_perpendicular[0] / right_eye_perpendicular_length, right_eye_perpendicular[1] / right_eye_perpendicular_length)
        
        # # Define the length of the lines perpendicular to the eyes
        # # line_length = 400
        
        # # Calculate the endpoints of the lines perpendicular to the eyes
        # left_eye_perpendicular_start = (left_eye_middle[0] - int(left_eye_perpendicular[0] * line_length), 
        #                                 left_eye_middle[1] - int(left_eye_perpendicular[1] * line_length))
        # left_eye_perpendicular_end = (left_eye_middle[0] + int(left_eye_perpendicular[0] * line_length), 
        #                               left_eye_middle[1] + int(left_eye_perpendicular[1] * line_length))
        
        # right_eye_perpendicular_start = (right_eye_middle[0] - int(right_eye_perpendicular[0] * line_length), 
        #                                  right_eye_middle[1] - int(right_eye_perpendicular[1] * line_length))
        # right_eye_perpendicular_end = (right_eye_middle[0] + int(right_eye_perpendicular[0] * line_length), 
        #                                right_eye_middle[1] + int(right_eye_perpendicular[1] * line_length))
        
        # Draw lines perpendicular to the eyes
        # cv2.line(image, left_eye_perpendicular_start, left_eye_perpendicular_end, (255, 0, 0), line_thickness)  # Left eye line
        # cv2.line(image, right_eye_perpendicular_start, right_eye_perpendicular_end, (255, 0, 0), line_thickness)  # Right eye line
        
        
        
        # Draw line through the leftmost and rightmost points of the lips
        leftmost_lip = (landmarks.part(48).x - line_length, landmarks.part(48).y)
        rightmost_lip = (landmarks.part(54).x + line_length, landmarks.part(54).y)
        cv2.line(image, leftmost_lip, rightmost_lip, (255, 0, 0), line_thickness)  # Lips line
        
        # Save the image with landmarks in the output folder
        output_path = os.path.join(output_folder, f"Landmark_{filename}")
        cv2.imwrite(output_path, image)
        print(f"Image with landmarks saved as '{output_path}'")
