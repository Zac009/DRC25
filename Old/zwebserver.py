from flask import Flask, Response, render_template_string
import cv2
import threading
import time

app = Flask(__name__)

# Global shared frames
frame1 = None
frame2 = None

# Lock to safely update/read frames across threads
lock = threading.Lock()

def video_capture_thread():
    global frame1, frame2
    cap = cv2.VideoCapture(1)  # Use 0 or 1 depending on your camera
    if not cap.isOpened():
        print("Camera failed to open")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Example: Simulated two masks (or processed frames)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        mask2 = cv2.Canny(frame, 100, 200)               # Edge detection
        mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)

        _, buffer1 = cv2.imencode('.jpg', mask1)
        _, buffer2 = cv2.imencode('.jpg', mask2)

        with lock:
            frame1 = buffer1.tobytes()
            frame2 = buffer2.tobytes()

        time.sleep(0.02)  # ~30 FPS

def generate_stream(mask_id):
    global frame1, frame2
    while True:
        with lock:
            if mask_id == 1 and frame1 is not None:
                frame = frame1
            elif mask_id == 2 and frame2 is not None:
                frame = frame2
            else:
                continue  # Skip if frame not ready

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    html = '''
    <html>
    <head>
        <style>
            .mask-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }
            .mask {
                flex: 1 1 45%;
            }
            img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ccc;
            }
        </style>
    </head>
    <body>
        <div class="mask-container">
            <div class="mask"><h3>Mask 1</h3><img src="/mask1" width="400" height="300"/></div>
            <div class="mask"><h3>Mask 2</h3><img src="/mask2" width="400" height="300"/></div>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/mask1')
def stream_mask1():
    return Response(generate_stream(1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mask2')
def stream_mask2():
    return Response(generate_stream(2), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=video_capture_thread, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5001, threaded=True)