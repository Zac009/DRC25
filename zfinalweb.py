from flask import Flask, Response, render_template_string
import cv2
import threading
import time
from zzVidImplementation import Vision

app = Flask(__name__)

# Global shared frames
frame1 = None
frame2 = None

# Lock to safely update/read frames across threads
lock = threading.Lock()
Ben = Vision(lock)

def generate_stream(mask_id):
    global frame1, frame2
    while True:
        with lock:
            frame = Ben.mask1 if mask_id == 1 else Ben.mask2
            if frame is None or not hasattr(frame, 'shape') or frame.size == 0:
                time.sleep(0.05)
                continue
            if len(frame.shape) == 2:  # grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.02)

@app.route('/')
def index():
    html = '''
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            background-color: #f7f7f7;
        }

        h3 {
            margin-bottom: 10px;
        }

        .mask-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 30px;
        }

        .mask {
            flex: 1 1 45%;
            background-color: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }

        .mask img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid #ddd;
        }

        .button-group {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
            margin-bottom: 20px;
        }

        button {
            padding: 14px 24px;
            font-size: 18px;
            border-radius: 10px;
            border: none;
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }

        button:hover {
            background-color: #45a049;
        }

        #status {
            font-weight: bold;
            color: #333;
        }
    </style>
</head>
<body>

    <div class="mask-container">
        <div class="mask">
            <h3>Mask 1</h3>
            <img src="/mask1" alt="Mask 1"/>
        </div>
        <div class="mask">
            <h3>Mask 2</h3>
            <img src="/mask2" alt="Mask 2"/>
        </div>
    </div>

    <div class="button-group">
        <button onclick="sendCommand('stop')">⏹️ Stop</button>
        <button onclick="sendCommand('play')">▶️ Play</button>
    </div>

    <p>Command Sent: <span id="status">None</span></p>

    <script>
        function sendCommand(command) {
            fetch('/command/' + command)
                .then(response => response.text())
                .then(data => {
                    document.getElementById('status').innerText = data;
                });
        }
    </script>

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

@app.route('/command/<cmd>')
def handle_command(cmd):
    if cmd == 'play':
        # Call your play function
        print("Play")
        if not Ben.running:
            threading.Thread(target=Ben.main()).start()
        return "Running"
    elif cmd == 'stop':
        print("Pause")
        # Call your stop function
        Ben.running = False
        return "Stopped"
    return cmd

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)