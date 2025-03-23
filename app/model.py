import os
import wandb
from loadotenv import load_env # removed in GCP deployment





MODELS_DIR = 'models'
MODEL_FILE_NAME = 'best_model.pth'

CATEGORIES = ["freshapple", "freshbanana", "freshorange", 
              "rottenapple", "rottenbanana", "rottenorange"]


load_env() # This will be removed for the GCP deployment
wandb_api_key = os.environ.get('WANDB_API_KEY')

MODELS_DIR = 'models'
MODEL_FILE_NAME = 'best_model.pth' # Take note that in other examples we called this model.pth

os.makedirs(MODELS_DIR, exist_ok=True)

assert 'WANDB_API_KEY' in os.environ, 'Please enter the wandb API key'

wandb_org = os.environ.get('WANDB_ORG')
wandb_project = os.environ.get('WANDB_PROJECT')
wandb_model_name = os.environ.get('WANDB_MODEL_NAME')
wandb_model_version = os.environ.get('WANDB_MODEL_VERSION')

artifact_path = f"{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}"

wandb.login(key=wandb_api_key)
artifact = wandb.Api().artifact(artifact_path, type='model')
artifact.download(root=MODELS_DIR)