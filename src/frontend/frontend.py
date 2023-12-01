import chainlit as cl
from llm.inference import Assistant

# Random id in die session reinschreiben

#assistant = Assistant()

@cl.on_message
async def main(message: cl.Message):
    '''Your custom logic goes here'''
    app_user = cl.user_session.get("user")
    await cl.Message(content=app_user).send()


    #response = assistant.answer(message.content)

    # Send a response back to the user
    await cl.Message(content=message.content).send()

if __name__ == '__main__':
    pass