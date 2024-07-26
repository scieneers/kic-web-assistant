# Dependencies
- python 3.11
- poetry 1.7.1 & `poetry install`
- task: `brew install go-task`
- install pre-commit hooks: [`pre-commit`](https://github.com/pre-commit/pre-commit) `install`

# Local development

## VectorDB [Qdrant](https://github.com/qdrant/qdrant-client)
`docker pull qdrant/qdrant:v1.6.1`
`docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant:v1.6.1`

`client = QdrantClient(host =QDRANT_URL, api_key=QDRANT_TOKEN, port=6333, grpc_port=6334 , https=False, prefer_grpc=True)`

## Run frontent: streamlit
Go into the src/frontend folder and run:
`streamlit run frontend.py`

## Docker

## How to build and run Image
`docker login` <br />
`docker build -t fatemeh001/kicampus_chatbot:0.0.1 .` <br />
`docker images` <br />
`docker run -p 8501 fatemeh001/kicampus_chatbot:0.0.1` <br />

`docker login kicwaacrdev.azurecr.io` <br />
`docker tag fatemeh001/kicamp_chatbot:0.0.1 kicwaacrdev.azurecr.io/fatemeh001/kicampus_chatbot:0.0.1` <br />
`docker push kicwaacrdev.azurecr.io/fatemeh001/kicampus_chatbot:0.0.1` <br />

## To run docker images locally, mount your credentials:
`docker run -it --rm -p 80:80 -v ~/.azure:/home/appuser/.azure kicwaacrdev.azurecr.io/rest-api:latest`

### Build and push Docker images locally
Before pushing the docker image you need to be authenticated docker via `gcloud auth configure-docker europe-west3-docker.pkg.dev`.

If you're working with a mac that is using an arm64 architecture, you specifically need to build a docker image based on an [amd architecture for cloud run](https://stackoverflow.com/questions/66920645/exec-format-error-when-running-containers-build-with-apple-m1-chip-arm-based).

# Data Extraction

# Moodle
To access content from Moodle, you need access to Moodle courses via the REST API. To set up the integration, do the following steps:
0. Get admin access to moodle.
1. Enable Web Services: _Site Administration_ -> _General_ -> _Advanced Features_ -> _Enable web services_
2. Enable REST Protocol: _Site Administration_ -> _Server_ -> _Web Services_ -> _Manage Protocols_ -> _Enable REST protocol_
3. (Optional): Create a technical new user/roles
4. Create a new external service _Site Administration_ -> _Server_ -> _External services_. Give it a name and enable _Enabled_, _Authorized users only_ and _Can download files_ (under _Show more..._).
5. Add the user as an _Authorised User_ to the external service.
6. Add the following functions to the external service:
    - core_block_get_course_blocks
    - core_course_get_categories
    - core_course_get_contents
    - core_course_get_course_content_items
    - core_course_get_course_module
    - core_course_get_courses
    - core_course_get_module
7. Create a token for the user and external service under _Site Administration_ -> _Server_ -> _Manage tokens_. This allows you to authenticate against the REST API.

You can try it out with a GET request against this url (swap TOKEN for your token und FUNCTION against the function to test):
https://ki-campus-test.fernuni-hagen.de/webservice/rest/server.php?wstoken=TOKEN&wsfunction=FUNCTION&moodlewsrestformat=json
