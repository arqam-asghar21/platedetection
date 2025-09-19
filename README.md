# License Plate Detection API

A FastAPI-based service that detects and anonymizes license plates in images using YOLOv8.

## API Endpoints

### GET /
Returns API information and available endpoints.

### GET /health
Health check endpoint.

### POST /detect
Detects and blurs license plates in uploaded images.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "image_base64": "base64_encoded_blurred_image",
  "detections": 2,
  "processing_time_ms": 150.25,
  "boxes": [
    {
      "xyxy": [x1, y1, x2, y2],
      "confidence": 0.85
    }
  ]
}
```

## Usage

1. Upload an image file to `/detect`
2. Receive the processed image with license plates blurred
3. The API returns detection count, processing time, and bounding boxes

## Features

- Supports various image formats (JPEG, PNG, HEIC, etc.)
- Automatic EXIF orientation correction
- Strong pixelation blur for license plate anonymization
- 50px minimum height blur coverage
- CORS enabled for web integration

## Deployment

This API is ready for deployment on Render, Heroku, or similar platforms.
