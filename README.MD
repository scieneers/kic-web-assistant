# Local development

## Install [Qdrant](https://github.com/qdrant/qdrant-client)
`docker pull qdrant/qdrant:v1.6.1`

`docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant:v1.6.1`

## Run frontent chainlit
`chainlit run src/frontend.py -w`