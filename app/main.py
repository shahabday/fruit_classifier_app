import torch 
import io 

from pydantic import BaseModel
from fastapi import FastAPI , UploadFile , File, Depends
from torchvision.models import ResNet
from app.model import load_model, load_transforms, CATEGORIES
from PIL import Image


# we need it for Swager and FastAPI tot work correctly
# This is a data model for the result
class Result(BaseModel):
    label : str
    probability : float


# Create the FastAPI instance

app = FastAPI()

@app.get('/')
def read_root():
    return {'message' : " call predict instead of root thi is an ML endpoint"}

@app.post('/predic', response_model=Result)
async def predict(

    input_image : UploadFile = File(...),
    model :  ResNet = Depends(load_model),
    transforms : transforms.Compose = Depends(load_transforms)

)

