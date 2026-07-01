def main():
    print("Hello from gesture-mqtt!")


if __name__ == "__main__":
    main()
    
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import paho.mqtt.client as mqtt

MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "Smarthome/Gestenerkennung/gestures/result"

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(MQTT_BROKER, MQTT_PORT, 60)

base_options = mp_tasks.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.HandLandmarker.create_from_options(options)


def fingers_closed(landmarks):
    """Prüft, ob Zeige-, Mittel-, Ring- und kleiner Finger eingeklappt sind."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    closed = 0
    for tip, pip in zip(tips, pips):
        if landmarks[tip].y > landmarks[pip].y:
            closed += 1
    return closed == 4


def detect_thumb_gesture(landmarks):
    if not fingers_closed(landmarks):
        return None

    thumb_tip = landmarks[4]
    wrist = landmarks[0]

    if thumb_tip.y < wrist.y - 0.1:
        return "thumb_up"
    elif thumb_tip.y > wrist.y + 0.1:
        return "thumb_down"
    return None


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera konnte nicht geöffnet werden")

last_gesture = None
frame_timestamp_ms = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_timestamp_ms += 33
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    gesture = None
    if result.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            gesture = detect_thumb_gesture(hand_landmarks)

    if gesture and gesture != last_gesture:
        client.publish(MQTT_TOPIC, gesture)
        print(f"Gesendet: {gesture}")
        last_gesture = gesture
    elif gesture is None:
        last_gesture = None

    cv2.putText(frame, f"Geste: {gesture or '-'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Thumb Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
client.disconnect()