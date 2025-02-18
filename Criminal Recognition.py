import cv2
import face_recognition
import os
from twilio.rest import Client

# Twilio Configuration
account_sid = 'Your_Account_sid'
auth_token = 'Your_Auth_Token'
twilio_phone_number = 'Your_Twilio_PN'
your_phone_number = 'Your_PN'

client = Client(account_sid, auth_token)

# Path to the folder containing criminal images
criminals_path = r"1. AI Camera/Criminal/Criminals File"

# Load all criminal images and encode their faces
criminal_encodings = []
criminal_names = []

for filename in os.listdir(criminals_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        criminal_image = face_recognition.load_image_file(os.path.join(criminals_path, filename))
        encoding = face_recognition.face_encodings(criminal_image)

        # Check if a face was detected and successfully encoded
        if len(encoding) > 0:
            criminal_encodings.append(encoding[0])
            criminal_names.append(os.path.splitext(filename)[0])

# Initialize the webcam
cap = cv2.VideoCapture(0)

alert_sent_for = None  # To track the last criminal who triggered an alert

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Detect face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    detected_criminal = None

    # Loop through each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face with the loaded criminal encodings
        matches = face_recognition.compare_faces(criminal_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the criminal's name
        if True in matches:
            match_index = matches.index(True)
            detected_criminal = criminal_names[match_index]
            name = detected_criminal

        # Draw a rectangle around the detected face and label it
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # If a new criminal is detected and hasn't been alerted yet
    if detected_criminal and detected_criminal != alert_sent_for:
        message_body = f"ALERT: Criminal Detected - {detected_criminal}!"
        try:
            message = client.messages.create(
                body=message_body,
                from_=twilio_phone_number,
                to=your_phone_number
            )
            print(f"SMS Alert Sent: {message_body}")
            alert_sent_for = detected_criminal  # Update the last alerted criminal
        except Exception as e:
            print(f"Failed to send SMS: {e}")

    # Reset the alert if no criminal is detected in the frame
    if not detected_criminal:
        alert_sent_for = None

    # Display the frame
    cv2.imshow("Criminal Face Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
