FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Install C compiler and set CC environment variable
RUN apt-get update && apt-get install -y gcc
ENV CC=/usr/bin/gcc
# Install g++ compiler and set CXX environment variable
RUN apt-get update && apt-get install -y g++
ENV CXX=/usr/bin/g++

# Set the working directory in the container
WORKDIR /app

# Make a directory
RUN mkdir /app/Legato
WORKDIR /app/Legato

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt

# Copy in the rest of the files and set up the package
COPY . /app/Legato
RUN pip install --no-cache-dir -e .
RUN ls -la

# Run the train.py script with wandb API key as a command-line argument
ENTRYPOINT ["python", "scripts/train.py"]