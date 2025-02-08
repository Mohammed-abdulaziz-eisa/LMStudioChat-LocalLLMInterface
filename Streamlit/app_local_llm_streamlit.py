import streamlit as st 
from openai import OpenAI

# Point to the local server 

client = OpenAI(base_url='http://localhost:1234/v1', api_key='lm-studio')

# set page title 
st.title('Chat with LM Studio')

# Initialize chat history in session state if it doesn't exist
if 'message' not in st.session_state:
    st.session_state.messages = [
        {'role': 'system', 'content': 'you are helpful assistant.'}
    ]
# Display chat input
user_input = st.chat_input('your message:')

# Display chat history and handle new inputs 
for message in st.session_state.messages:
    if message['role'] != 'system':
        with st.chat_message(message['role']):
            st.write(message['content'])

if user_input:
    # Display user message
    with  st.chat_message('user'):
        st.write(user_input)
    
    # Add user message to history 
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Get streaming response 
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        full_response = ""
                   

        completion = client.chat.completions.create(
            model='llama-3.2-3b-instruct',
            messages=st.session_state.messages,
            temperature=0.7,
            stream=True
        )
        
        # Process the streaming response 
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.write(full_response + ".... ")
        
        message_placeholder.write(full_response)
        
    # Add assistant response to history 
    st.session_state.messages.append({"role": "assistant", "content": full_response})
