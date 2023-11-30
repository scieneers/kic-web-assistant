from src.embedder.multilingual_e5_large import MultilingualE5LargeEmbedder
from src.vectordb.qdrant import VectorDBQdrant
from openai import AzureOpenAI
from pathlib import Path
import json

system_prompt = '''You are a chat bot on our website ki-campus.org helping and answering its user. We are a learning platform for artificial intelligence with free online courses, videos and podcasts in various topics of AI and data literacy. 
As an research and development project, the AI Campus is funded by the German Federal Ministry of Education and Research (BMBF). 
You audience speaks german or english, answer in the language the user asked the question in.

You will be provided with a pre-selection of context to answer the question. You don't need to use all of it, choose what you think is relevant. 
Dont add any additional information that is not in the context. Don't make up an answer! If you don't know the answer, just say that you don't know. 

Keep your answers short and simple, you are a chat bot!'''

user_question = '''
QUESTION: {question}
=========
CONTEXT: 
{context}
=========
'''

prompt=[
    {'role': 'system', 'content': system_prompt},
    #{'role': 'user', 'content': user_question_1},
    #{'role': 'assistant', 'content': assistant_answer_1},
    #{'role': 'user', 'content': user_question_2},
]

# get query
query = 'Welche Kurse zu ethischer KI habt ihr im Angebot?'

# embed query
embedder = MultilingualE5LargeEmbedder()
embedding = embedder.embed(query, type='query')

# retrieve context
vectordb = VectorDBQdrant()
data_points = vectordb.search(collection_name='test_collection', query_vector=embedding)
context = [point.payload['vector_content'] for point in data_points]

# llm 
with open(Path(__file__).parent / 'keys.json', 'r') as f:
    keys = json.load(f)

client = AzureOpenAI(api_key=keys['AZURE_OPENAI_KEY'], api_version='2023-05-15', azure_endpoint=keys['AZURE_OPENAI_ENDPOINT'])

## create prompt
prompt.append({'role': 'user', 'content': user_question.format(question=query, context='\n\n'.join(context))})

response = client.chat.completions.create(
    model='gpt-3_5', # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
    messages=prompt,
    temperature=0.8,
    stream=False)

print(response.choices[0].message.content)

print('wait')

