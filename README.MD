# Dependencies
python 3.11
poetry 1.7.1 
task: `brew install go-task`

# Local development

## Install [Qdrant](https://github.com/qdrant/qdrant-client)
`docker pull qdrant/qdrant:v1.6.1`
`docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant:v1.6.1`

## Run frontent: streamlit
Go into the src/frontend folder and run:
`streamlit run frontend.py `


## Docker
### Build and push Docker images locally
Before pushing the docker image you need to be authenticated docker via `gcloud auth configure-docker europe-west3-docker.pkg.dev`.

If you're working with a mac that is using an arm64 architecture, you specifically need to build a docker image based on an [amd architecture for cloud run](https://stackoverflow.com/questions/66920645/exec-format-error-when-running-containers-build-with-apple-m1-chip-arm-based).
