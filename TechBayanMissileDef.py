#TechBayanMissileDef.py
# Author: leyritana.jun.se@icloud.com   
# Date Created: 14th July 2025
# Description: This script implements a Flask web application for missile detection and tracking using YOLOv8 and DeepSort with RDF generation capabilities. WGS84 geospatial data is also utilized.
# Add-ons: Flight path visualization embedded in the map view, user GPS center setting as the refernce point for missile trajectory plotting
# Disclaimer: This code is for educational purposes only and should not be used for any malicious activities.
# Prerequisites: Install the required libraries using pip:
# pip install ultralytics torch opencv-python flask rdflib geopy deep_sort_realtime


from flask import Flask, Response, request, render_template_string, redirect, jsonify
import os
import time
import math
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from rdflib import Graph, Literal, RDF, URIRef, Namespace, XSD
from geopy.distance import geodesic
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('static', exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

user_gps_center = None
detection_active = False
show_paths = True
cap = None
current_frame = None
trajectories = {}

detector = YOLO("yolov8s.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tracker = DeepSort(max_age=30, n_init=3)

HTML = '''
<html><head><title>TechBayan Missile Tracker</title></head><body>
<h1>TechBayan Missile Detection & Tracking</h1>
<form method="POST" action="/select_model" enctype="multipart/form-data">
    <label>Upload YOLOv8 model file (.pt):</label>
    <input type="file" name="model_file" accept=".pt">
    <button type="submit">Load Model</button>
</form><br>
<img src="/video_feed" width="960"/><br><br>
<form method="POST" action="/toggle"><button>{{ 'Stop' if active else 'Start' }} Detection</button></form>
<form method="POST" action="/snapshot"><button>Snapshot</button></form>
<form method="POST" action="/toggle_path"><button>{{ 'Hide' if show_paths else 'Show' }} Flight Path</button></form>
<form method="POST" action="/generate_rdf">
    <label>Missile Origin Country:</label>
    <input type="text" name="originCountry" placeholder="e.g. Russia" required><br>
    <label>Missile Type:</label>
    <input type="text" name="missileType" placeholder="e.g. Cruise" required><br>
    <button type="submit">Generate RDF (Geo)</button>
</form>
<form method="GET" action="/map"><button>View Map</button></form>
</body></html>
'''

def pixel_to_latlon(x, y, width, height):
    global user_gps_center
    if not user_gps_center:
        return 0.0, 0.0
    center_lat, center_lon = user_gps_center
    scale_lat = 0.00005
    scale_lon = 0.00005
    lat = center_lat + (y - height / 2) * scale_lat
    lon = center_lon + (x - width / 2) * scale_lon
    return round(lat, 6), round(lon, 6)

def generate_rdf(current_frame, trajectories, pixel_to_latlon_func, origin_country, missile_type):
    g = Graph()
    MSL = Namespace("http://techbayan.org/missile/")
    GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
    TIME = Namespace("http://www.w3.org/2006/time#")
    g.bind("msl", MSL)
    g.bind("geo", GEO)
    g.bind("time", TIME)

    frame_h, frame_w = current_frame.shape[0], current_frame.shape[1]

    for tid, pts in trajectories.items():
        track_uri = URIRef(MSL[f"track/{tid}"])
        g.add((track_uri, RDF.type, MSL.Missile))
        g.add((track_uri, MSL.isThreatLevel, Literal("high")))
        g.add((track_uri, MSL.detectedBy, Literal("YOLOv8s + DeepSort")))
        g.add((track_uri, MSL.originCountry, Literal(origin_country)))
        g.add((track_uri, MSL.missileType, Literal(missile_type)))

        for i, (x, y) in enumerate(pts):
            lat, lon = pixel_to_latlon_func(x, y, frame_w, frame_h)
            timestamp = int(time.time())
            point_uri = URIRef(MSL[f"track/{tid}/point/{i}"])
            g.add((track_uri, MSL.hasPoint, point_uri))
            g.add((point_uri, RDF.type, GEO.Point))
            g.add((point_uri, GEO.lat, Literal(lat, datatype=XSD.decimal)))
            g.add((point_uri, GEO.long, Literal(lon, datatype=XSD.decimal)))
            g.add((point_uri, MSL.pixelX, Literal(x, datatype=XSD.integer)))
            g.add((point_uri, MSL.pixelY, Literal(y, datatype=XSD.integer)))
            g.add((point_uri, TIME.inXSDDateTimeStamp, Literal(timestamp, datatype=XSD.integer)))

            if i > 0:
                prev_x, prev_y = pts[i - 1]
                prev_lat, prev_lon = pixel_to_latlon_func(prev_x, prev_y, frame_w, frame_h)
                distance = geodesic((prev_lat, prev_lon), (lat, lon)).meters
                speed = distance / 1
                g.add((point_uri, MSL.hasSpeed, Literal(round(speed, 2), datatype=XSD.decimal)))

    rdf_filename = f"static/traj_{int(time.time())}_enriched.rdf"
    g.serialize(destination=rdf_filename, format='xml')
    rdf_turtle = g.serialize(format='turtle').encode("utf-8")
    return rdf_filename, rdf_turtle

@app.route('/')
def index():
    return render_template_string(HTML, active=detection_active, show_paths=show_paths)

@app.route('/select_model', methods=['POST'])
def select_model():
    global detector
    model_file = request.files.get('model_file')
    if model_file and model_file.filename.endswith('.pt'):
        path = os.path.join(app.config['UPLOAD_FOLDER'], model_file.filename)
        model_file.save(path)
        detector = YOLO(path)
    return redirect('/')

@app.route('/toggle', methods=['POST'])
def toggle():
    global detection_active
    detection_active = not detection_active
    return redirect('/')

@app.route('/toggle_path', methods=['POST'])
def toggle_path():
    global show_paths
    show_paths = not show_paths
    return redirect('/')

@app.route('/snapshot', methods=['POST'])
def snapshot():
    if current_frame is not None:
        filename = f"snapshot_{int(time.time())}.jpg"
        path = os.path.join('static', filename)
        cv2.imwrite(path, current_frame)
        return f"<html><body><h3>Snapshot Saved</h3><img src='/static/{filename}' width='800'/><br><a href='/'>Back</a></body></html>"
    return redirect('/')

@app.route('/generate_rdf', methods=['POST'])
def rdf():
    if current_frame is None:
        return redirect('/')
    origin_country = request.form.get("originCountry", "Unknown")
    missile_type = request.form.get("missileType", "Unknown")
    rdf_filename, rdf_turtle = generate_rdf(current_frame, trajectories, pixel_to_latlon, origin_country, missile_type)
    return f'''
    <html><head><title>Generated RDF</title></head><body>
    <h3>RDF Data (Turtle Format)</h3>
    <pre style="background-color:#f4f4f4; padding:1em; overflow:auto; max-height:500px;">{rdf_turtle}</pre>
    <a href='/{rdf_filename}' download>Download RDF File</a><br>
    <a href='/'>â¬ Back to Home</a>
    </body></html>
    '''

@app.route('/set_reference_location')
def set_reference_location():
    global user_gps_center
    lat = float(request.args.get('lat', 0))
    lon = float(request.args.get('lon', 0))
    user_gps_center = (lat, lon)
    return '', 204

@app.route('/trajectory_geojson')
def trajectory_geojson():
    features = []
    if current_frame is None:
        return jsonify({"type": "FeatureCollection", "features": []})
    frame_w, frame_h = current_frame.shape[1], current_frame.shape[0]
    for tid, pts in trajectories.items():
        coords = [[pixel_to_latlon(x, y, frame_w, frame_h)[1], pixel_to_latlon(x, y, frame_w, frame_h)[0]] for x, y in pts]
        if coords:
            features.append({
                "type": "Feature",
                "properties": {"id": tid},
                "geometry": {"type": "LineString", "coordinates": coords}
            })
    return jsonify({"type": "FeatureCollection", "features": features})

@app.route('/map')
def map_view():
    return '''
    <html>
    <head>
        <title>Trajectory Map</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
        <style>#map { height: 100vh; }</style>
    </head>
    <body>
    <div id="map"></div>

    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([0, 0], 17);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);

        var userMarker = L.marker([0, 0]).addTo(map);

        navigator.geolocation.getCurrentPosition(function(pos) {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;

            userMarker.setLatLng([lat, lon]);
            map.setView([lat, lon], 17);
            fetch(`/set_reference_location?lat=${lat}&lon=${lon}`);

            // Show popup after centering
            userMarker.bindPopup("📍 You are here. Missile trajectories are now plotted relative to this location.").openPopup();
        });

        fetch('/trajectory_geojson')
          .then(res => res.json())
          .then(data => {
              L.geoJSON(data, { style: { color: 'red', weight: 3 } }).addTo(map);
          });
    </script>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global cap, current_frame
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        current_frame = frame.copy()
        if detection_active:
            results = detector.predict(source=frame, conf=0.25, device=device, verbose=False)
            detections = []
            for r in results:
                for i, box in enumerate(r.boxes.xyxy):
                    x1, y1, x2, y2 = [int(coord.item()) for coord in box]
                    conf = float(r.boxes.conf[i].item())
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "missile"))
            tracks = tracker.update_tracks(detections, frame=frame)
            for tr in tracks:
                if not tr.is_confirmed(): continue
                tid = tr.track_id
                x1, y1, x2, y2 = map(int, tr.to_ltrb())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trajectories.setdefault(tid, []).append((cx, cy))
                
                if show_paths and len(trajectories[tid]) > 1:
                    for i in range(1, len(trajectories[tid])):
                        cv2.line(frame, trajectories[tid][i - 1], trajectories[tid][i], (0, 255, 255), 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)



        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True, port=5000)

@app.route('/toggle_path', methods=['POST'])
def toggle_path():
    global show_paths
    show_paths = not show_paths
    return redirect('/')