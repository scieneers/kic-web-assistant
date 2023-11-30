import chainlit as cl
from src.inference import Assistant

assistant = Assistant()

@cl.on_message
async def main(message: cl.Message):
    '''Your custom logic goes here'''

    response = assistant.answer(message.content)

    # Send a response back to the user
    await cl.Message(content=response).send()
