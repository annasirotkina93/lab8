import cv2
import numpy as np

def track_marker_with_area():
    cap = cv2.VideoCapture(0)
    down_points = (640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        h, w = frame.shape[:2]
        mid_x = w // 2

        cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(
            mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)

            x = int(x)
            y = int(y)
            center = (x, y)
            radius = int(radius)

            if x > mid_x:
                color = (0, 255, 0)
                position_text = "Right side"
            else:
                color = (0, 0, 255)
                position_text = "Left side"

            cv2.circle(frame, center, radius, color, 2)
            cv2.putText(frame, position_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Marker Tracking with Area Check', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    track_marker_with_area()