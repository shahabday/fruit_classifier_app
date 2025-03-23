import torch 
import io 

from pydantic import BaseModel
from fastapi import FastAPI , UploadFile , File, Depends
from torchvision.models import ResNet
from app.model import load_model, load_transforms, LABELS
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

@app.post('/predict', response_model=Result)
async def predict(
        input_image: UploadFile = File(...),
        model: ResNet = Depends(load_model),
        transforms: transforms.Compose = Depends(load_transforms)
) -> Result:
    image = Image.open(io.BytesIO(await input_image.read()))

    # Here we delete the alpha channel, the model doesn't use it
    # and will complain if the input has it     
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Here we add a batch dimension of 1 
    image = transforms(image).unsqueeze(0)

    # This is inference mode, we don't need gradients 
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)


    label = LABELS[predicted_class.item()]

    return Result(label=label, probability=confidence.item())
