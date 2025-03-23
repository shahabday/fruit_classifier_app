# Set Python version for the image 
FROM python:3.12-slim

#set the working dierctory in the container 
WORKDIR /code

# Copy the dependencies file to the Working dierctory
COPY ./requirements.txt /code/requirements.txt

# Run and install the dependencies
RUN pip install -r /code/requirements.txt

# copy the content of the local app directory to the wdirecotry 

COPY ./app /code/app

# Set Env variables

ENV WANDB_API_KEY=""
ENV WANDB_ORG=""
ENV WANDB_PROJECT=""
ENV WANDB_MODEL_NAME=""
ENV WANDB_MODEL_VERSION=""



EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]