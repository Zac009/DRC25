# app.py
from flask import Flask, Response, render_template_string
import cv2
import videoImplementation


app = Flask(__name__)

def generate_mask_stream(mask_func):
    cap = cv2.VideoCapture('qut_demo.mov')  # Use 0 for default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = mask_func(frame)

        # Convert single channel to BGR for JPEG encoding
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        _, buffer = cv2.imencode('.jpg', mask)
        frame = buffer.tobytes()
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
            <div class="mask"><h3>Mask 3</h3><img src="/mask3" width="400" height="300"/></div>
            <div class="mask"><h3>Mask 4</h3><img src="/mask4" width="400" height="300"/></div>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/mask1')
def stream_mask1():
    return Response(generate_mask_stream(videoImplementation.combined_mask),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mask2')
def stream_mask2():
    return Response(generate_mask_stream(videoImplementation.combined_mask),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mask3')
def stream_mask3():
    return Response(generate_mask_stream(videoImplementation.combined_mask),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mask4')
def stream_mask4():
    return Response(generate_mask_stream(videoImplementation.combined_mask),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/controls')
def control():
    html = '''
    <html>
    <head>
        <title>Robot Controls</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                margin-top: 50px;
            }
            .grid {
                display: grid;
                grid-template-columns: 100px 100px 100px;
                grid-gap: 10px;
                justify-content: center;
                margin-bottom: 20px;
            }
            button {
                padding: 20px;
                font-size: 18px;
                border-radius: 10px;
                border: none;
                background-color: #4CAF50;
                color: white;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>Robot Control Panel</h1>
        <div class="grid">
            <div></div>
            <button onclick="sendCommand('forward')">↑</button>
            <div></div>

            <button onclick="sendCommand('left')">←</button>
            <button onclick="sendCommand('stop')">⏹️</button>
            <button onclick="sendCommand('right')">→</button>

            <div></div>
            <button onclick="sendCommand('backward')">↓</button>
            <div></div>
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

if __name__ == '__main__':
    #app.run(host='172.20.10.2', port=5000, threaded=True)
    app.run(host='0.0.0.0', port=5001, threaded=True)
