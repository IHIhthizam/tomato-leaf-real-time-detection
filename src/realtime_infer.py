import os
import logging
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ========== CONFIGURATION ==========

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Confidence threshold for rejection
CONF_THRESHOLD = 0.5

# Green color range for leaf masking (HSV)
LOWER_GREEN = np.array([25, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])

# Class labels
class_labels = {
    0: "Tomato Bacterial Spot",
    1: "Tomato Early Blight",
    2: "Tomato Late Blight",
    3: "Tomato Leaf Mold",
    4: "Tomato Septoria Leaf Spot",
    5: "Tomato Spider Mites (Two Spotted Spider Mite)",
    6: "Tomato Target Spot",
    7: "Tomato Tomato Yellow Leaf Curl Virus",
    8: "Tomato Mosaic Virus",
    9: "Tomato Healthy"
}

# ========== LOAD MODEL ==========
try:
    model = load_model('tomate_model.keras')
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    exit()

# ========== START VIDEO CAPTURE ==========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("❌ Webcam could not be opened.")
    exit()

logger.info("🚀 Starting real-time tomato leaf disease detection. Press 'q' to exit.")

# ========== MAIN LOOP ==========
while True:
    ret, frame = cap.read()
    if not ret:
        logger.error("❌ Failed to read frame from webcam.")
        break

    try:
        # Convert to HSV for green mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Select largest green area
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # Ignore small areas (noise)
            if w * h > 5000:
                leaf_region = frame[y:y+h, x:x+w]
                img = cv2.resize(leaf_region, (224, 224))
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=0)

                # Predict
                prediction = model.predict(img, verbose=0)[0]
                max_prob = np.max(prediction)
                predicted_class_idx = np.argmax(prediction)
                confidence_score = round(max_prob * 100, 2)

                # Apply threshold
                if max_prob < CONF_THRESHOLD:
                    predicted_label = "Unknown"
                else:
                    predicted_label = class_labels[predicted_class_idx]

                text = f"{predicted_label} ({confidence_score}%)"
                logger.info(f"✅ Prediction: {text}")

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No significant leaf detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No leaf detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Display
        cv2.imshow("ToMate - Real Time Detector", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("🛑 Exiting real-time detection.")
            break

    except Exception as e:
        logger.error(f"❌ Error during frame processing: {e}")
        continue

# ========== CLEANUP ==========
cap.release()
cv2.destroyAllWindows()
