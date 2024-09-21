from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import base64,io
from PIL import Image
app = FastAPI()
from ultralytics import YOLO
model = YOLO("yolov8n.pt")


# Define a root route
@app.get("/")
async def read_root():
    return {"message": "Welcome to my FastAPI app!"}


# Define another route that takes a path parameter
@app.post("/detect_objects/")
async def detect_objects(image_data:dict):
    # Decode the base64 image
    images_list = image_data.get("images")
    print("Detection Started!")
    detected_images= []
    for item in images_list:
        image_bytes = base64.b64decode(item.get("base64"))
        image = Image.open(io.BytesIO(image_bytes))
        results = model.predict(image, save=False)
        #coordinates = results[0].boxes.xyxy.numpy().tolist()
        #classes = results[0].boxes.cls.numpy().tolist()
        #conf = results[0].boxes.conf.numpy().tolist()
        annotated_image = results[0].plot()
        buffered = io.BytesIO()
        annotated_image = Image.fromarray(annotated_image)
        if annotated_image.mode == "RGB":
            r, g, b = annotated_image.split()
            annotated_image = Image.merge("RGB", (b, g, r))

        annotated_image.save(buffered, format="JPEG")
        annotated_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        detected_images.append({"base64":annotated_image_base64})
    return JSONResponse(content={"images":detected_images })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
