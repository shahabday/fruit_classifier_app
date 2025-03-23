import os
import wandb
from loadotenv import load_env # removed in GCP deployment

from torchvision.models import resnet18,ResNet
from torch import nn


MODELS_DIR = 'models'
MODEL_FILE_NAME = 'best_model.pth'

LABELS  = ["freshapple", "freshbanana", "freshorange", 
              "rottenapple", "rottenbanana", "rottenorange"]


load_env() # This will be removed for the GCP deployment
wandb_api_key = os.environ.get('WANDB_API_KEY')

MODELS_DIR = 'models'
MODEL_FILE_NAME = 'best_model.pth' # Take note that in other examples we called this model.pth

os.makedirs(MODELS_DIR, exist_ok=True)

def download_artifact():
    assert 'WANDB_API_KEY' in os.environ, 'Please enter the wandb API key'

    wandb_org = os.environ.get('WANDB_ORG')
    wandb_project = os.environ.get('WANDB_PROJECT')
    wandb_model_name = os.environ.get('WANDB_MODEL_NAME')
    wandb_model_version = os.environ.get('WANDB_MODEL_VERSION')

    artifact_path = f"{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}"

    wandb.login(key=wandb_api_key)
    artifact = wandb.Api().artifact(artifact_path, type='model')
    artifact.download(root=MODELS_DIR)

def get_raw_model() -> ResNet  :
    architucture = resnet18(weights = None)
    architucture.fc = nn.Sequential (
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512,6)
    )
    return architucture

def load_model() -> ResNet:
    """This returns the model with its wandb weights"""
    download_artifact()
    model = get_raw_model()
    # Get the trained model weights
    model_state_dict_path = Path(MODELS_DIR) / MODEL_FILE_NAME
    model_state_dict = torch.load(model_state_dict_path, map_location='cpu')
    # Assign the trained model weights to model, this will fail for incomplete files 
    # Check the file size on wandb.ai, the resnet18 artifact should have 45.8 MB in size
    model.load_state_dict(model_state_dict, strict=True)
    # Turn off BatchNorm and Dropout
    model.eval()
    return model


def load_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])