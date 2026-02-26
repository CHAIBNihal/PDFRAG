
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from pdfProccess import get_pdf_text, create_chunks, get_vectores, get_conversation_chain
from TemplateHtml import css, bot_template, user_template
def handle_user_input(user_question):
       response = st.session_state.conversation.invoke({"question": user_question, 
                        "chat_history": st.session_state.chat_history
                        })
       st.session_state.chat_history.append(HumanMessage(content=user_question))
       st.session_state.chat_history.append(AIMessage(content=response))
       for i, message in enumerate(st.session_state.chat_history) :
          if i % 2 == 0 :
             st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
          else :
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
       #st.write(response)


def main():
   load_dotenv()
   st.write(css, unsafe_allow_html=True)
   if "conversation" not in st.session_state :
      st.session_state.conversation = None
   if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
   
   st.set_page_config(page_title="My IA Agent", page_icon=":books:")
   st.header("IA Agent - master PDF reader")
   user_question = st.text_input("Ak Something about your PDF here")
   if user_question:
      handle_user_input(user_question)
   

   with st.sidebar : 
       st.subheader('You PDFs documents')
       pdf_docs = st.file_uploader("Upload your PDF files here and click here to lanch processing", accept_multiple_files=True, type=["pdf"])
       if st.button("Process PDF") :
           
           with st.spinner("Processing...") :
            print("Processing PDF files...")   
            # Recuperer le text de pdf   
            raw_text =  get_pdf_text(pdf_docs)


            # convertir le text en chunks
            text_chunks = create_chunks(raw_text)
            
            
            # generer les embeddings (vectorestore) pour les chunks
            vectores =  get_vectores(text_chunks)


            # create conversation chain
            conversation = get_conversation_chain(vectores)
            #st.write(conversation)
            st.session_state.conversation = conversation
            st.session_state.chat_history = []
    

if __name__ == "__main__":
    main()