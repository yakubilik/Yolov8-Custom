# YOLOv8 Custom Object Detection API

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A lightweight REST API for real-time object detection using Ultralytics YOLOv8. Send base64-encoded images to the `/detect_objects/` endpoint and receive annotated images with bounding boxes drawn around detected objects.

## Features

- Real-time object detection powered by YOLOv8
- RESTful API built with FastAPI
- Accepts and returns base64-encoded images
- Batch processing support (multiple images per request)
- Annotated output with bounding boxes and class labels
- Auto-generated interactive API docs (Swagger UI)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Model | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| API Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| ASGI Server | [Uvicorn](https://www.uvicorn.org/) |
| Image Processing | [Pillow](https://python-pillow.org/) |

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yakubilik/Yolov8-Custom.git
   cd Yolov8-Custom
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Start the server

```bash
python main.py
```

The API server starts at `http://0.0.0.0:8080`. Hot-reload is enabled by default for development.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check / welcome message |
| `POST` | `/detect_objects/` | Run object detection on one or more images |

### Interactive Docs

Once the server is running, visit:

- **Swagger UI** -- [http://localhost:8080/docs](http://localhost:8080/docs)
- **ReDoc** -- [http://localhost:8080/redoc](http://localhost:8080/redoc)

### Example Request

Send a POST request with base64-encoded images:

```json
{
  "images": [
    { "base64": "<base64-encoded-image-string>" }
  ]
}
```

**Response:**

```json
{
  "images": [
    { "base64": "<base64-encoded-annotated-image>" }
  ]
}
```

### Example with cURL

```bash
curl -X POST http://localhost:8080/detect_objects/ \
  -H "Content-Type: application/json" \
  -d '{"images": [{"base64": "'$(base64 -w 0 sample.jpg)'"}]}'
```

## Training with a Custom Model

By default, the API loads the pretrained `yolov8n.pt` model. To use your own custom-trained model:

1. Train a YOLOv8 model on your dataset (see the [Ultralytics training guide](https://docs.ultralytics.com/modes/train/)).
2. Replace the model path in `main.py`:

   ```python
   model = YOLO("path/to/your/best.pt")
   ```

3. Restart the server.

## Project Structure

```
Yolov8-Custom/
├── main.py              # FastAPI application with detection endpoint
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
├── LICENSE              # MIT License
└── README.md            # Project documentation
```

## License

This project is licensed under the [MIT License](LICENSE).
