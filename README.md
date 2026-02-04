# SLM_prototype


docker build -t slm-model .

docker run --rm --gpus all --name loop-slm-model-services  -p 8002:8000 slm-model

docker run -d --gpus all --name loop-slm-model-services --restart unless-stopped -p 8002:8000  slm-model
