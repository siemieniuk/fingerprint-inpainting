import base64
from io import BytesIO

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from data_scripts.data_preprocessing import normalize_pixels, resize
from ml.models import get_unet
from ml.tuners import tune_model_mse
from schemas import ImageModel

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = dict()

# Values from the experiments
activation = "elu"
depth = 3
dropout = 0.07745662128302169

# Load model
model = get_unet(depth, dropout, activation)
model.load_weights("./weights/model_mse.15.h5")


@app.post("/api/v1/unet")
async def process_mobile_sam_onnx(image: ImageModel):
    """
    Input:
    - 'model' - name of the model (custom_1, custom_2, unet, unet_gan)
    Output:
    - 'image' - base64 string containing JPEG content
    """
    image_bytes = image.image_bytes

    # Convert image bytes to image
    image = Image.open(BytesIO(base64.b64decode(image_bytes)))
    image = np.array(image)
    image = resize(image)
    image = normalize_pixels(image)

    result = model.predict(image[np.newaxis])[0]
    result = ((1 - result) * 255.0).astype(np.uint8)
    result = np.repeat(result, 3, axis=-1)
    result = Image.fromarray(result)

    buffered = BytesIO()
    result.save(buffered, format="JPEG")
    ret_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = ImageModel(image_bytes=ret_image)
    return response
