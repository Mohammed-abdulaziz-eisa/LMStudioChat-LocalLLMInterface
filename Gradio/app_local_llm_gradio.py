import gradio as gr 
from openai import OpenAI

# Point to local server
client = OpenAI(base_url = 'http://localhost:1234/v1', api_key= 'lm-studio')

def generate_response(message, history):
    # convert history from tuples to message format
    messages=[
        {'role':'system', 'content': 'you are a helpful assistant.'}
    ]
    
    # add history messages
    for user_msg, assistant_msg in history:
        messages.append({'role': 'user', 'content': user_msg})
        if assistant_msg:          # Only add assistant message if exists
            messages.append({'role':'assistant', 'content': assistant_msg})
    
    # Add current message
    messages.append({'role': 'user', 'content': message})
    
    # create streaming completion 
    completion = client.chat.completions.create(
        model= 'llama-3.2-3b-instruct',
        messages = messages,
        temperature = 0.7,
        stream = True
    )
    
    # Process the streaming response 
    partial_message = ""
    for chunk in completion:
        # check for content in the delta
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            # Handle <think> and </think> tags
            content = content.replace("<think>", "Streaming................\n").replace("</think>", "\n\n Answer: ")
            partial_message += content
            yield partial_message
    

# create the Gradio interface with Blocks
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def bot(history):
        history[-1][1] = ""
        for chunk in generate_response(history[-1][0], history[:-1]):
            history[-1][1] = chunk
            yield history 
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue= False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    

if __name__ == '__main__':
    demo.launch()