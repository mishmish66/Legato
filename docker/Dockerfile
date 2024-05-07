FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install Git and other dependencies
RUN apt-get update && apt-get install -y git

# Clone your GitHub repository
RUN git clone https://github.com/mishmish66/Legato.git /app/Legato
WORKDIR /app/Legato

# Install the required packages from requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt
RUN pip install --no-cache-dir -e .

# Run the train.py script with wandb API key as a command-line argument
CMD ["sh", "-c", "wandb login $1 && python scripts/train.py"]
