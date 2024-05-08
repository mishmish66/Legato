# Legato
## Docker Building Instructions
To build the Docker image, run the following commands in the terminal:
```bash
docker build -t legato .  # Build the Docker image
docker tag legato mlvovsky/legato:latest  # Tag the Docker image
docker push mlvovsky/legato:latest  # Push the Docker image to Docker Hub
```


## Training Run Instructions
### Args
- `--wandb_api_key` (optional): Specifies the API key to be used. If provided, it will use the specified key. Otherwise, it will default to the one that is signed in before.
- `--data` (optional): Specifies the data directory. If provided, it will use the specified directory. Otherwise, it will default to `data.npz`.
- `--data_url` (optional): Specifies the data URL. If provided, it will download the data from the specified URL. Otherwise, it will just use the local data in `data.npz`.
### Running Locally
```bash
python train.py
```
### Running in Docker
```bash
docker run -it --runtime=nvidia mlvovsky/legato:latest
```