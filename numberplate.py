import cv2
import easyocr
import imutils

# Load Haar Cascade for number plate
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Load Video
video = cv2.VideoCapture('video.mp4')

# To store unique detected license plates
unique_plates = set()

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect number plates
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 40))

    for (x, y, w, h) in plates:
        roi = frame[y:y+h, x:x+w]
        text = reader.readtext(roi)

        for detection in text:
            plate_text = detection[1].strip()

            # Only add to set if it's a likely valid plate (length check avoids junk)
            if len(plate_text) >= 5:
                unique_plates.add(plate_text)

            # Draw bounding box and show plate text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Number Plate Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Show total number of unique vehicles detected
print(f"\nTotal Unique Vehicles Detected: {len(unique_plates)}")
print("Detected Plates:", unique_plates)

video.release()
cv2.destroyAllWindows()
