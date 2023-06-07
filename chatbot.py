import os
import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.vectorstores import Pinecone
import pinecone

def GetQuestion( _query, _memory):
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Question should be in Polish. 
    Do not repeat the question from the conversation.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question in Polish:"""

    Q_PROMPT = PromptTemplate.from_template(_template)
    chain = LLMChain(llm=QUESTION_LLM, memory=_memory, prompt=Q_PROMPT)
    output = chain.predict(question=_query)
    return output

def GetAnswer(_query:str, vectorstore,_search_elements=6):

    docs = vectorstore.similarity_search(_query, k=_search_elements)
    total_words = 0
    for i in range(len(docs)):
        total_words += len(docs[i].page_content.split())
        if total_words > MAX_CONTEXT_WORDS:
            docs = docs[:i]
            break

    prompt_template_p = """ Użyj poniższego kontekstu do wygenerowania wyczerpującej odpowiedzi na końcu. Po podaniu odpowiedzi zasugeruj zbliżone zagadnienia zgodne z kontakstem.
                    Jeżeli nie znasz odpowiedzi odpowiedz Nie wiem, nie staraj się wymyślić odpowiedzi.
                    {context}
                    Pytanie: {question}
                    Odpowiedź:"""
    prompt_template_p2 = """ Jesteś botem udzielającym informacji przedsiębiorcom na temat prowadzenia biznesu.
                    Odpowiedzi powinny być uprzejme i napisane prostym językiem.
                    Użyj poniższego kontekstu do wygenerowania wyczerpującej odpowiedzi na końcu. 
                    Jeżeli brakuje ci informacji -  dopytaj.
                    Po podaniu odpowiedzi zasugeruj zbliżone zagadnienia zgodne z kontekstem.
                    Jeżeli nie znasz odpowiedzi odpowiedz Nie wiem, nie staraj się wymyślić odpowiedzi.
                    {context}
                    Pytanie: {question}
                    Odpowiedź:"""


    PROMPT = PromptTemplate(
        template=prompt_template_p2, input_variables=["context", "question"]
        )

    chain = load_qa_chain(ANSWER_LLM, chain_type="stuff", prompt=PROMPT,verbose=False)
    output = chain({"input_documents": docs, "question": _query}, return_only_outputs=False)
    output_text = output["output_text"]+"\nZrodla:\n"
    for doc in output["input_documents"]:
        output_text += f'[{len(doc.page_content.split())}], {doc.metadata["source"]} \n'

    return output["output_text"] , output_text

pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],  # find at app.pinecone.io
        environment="us-east4-gcp"  # next to api key in console
    )
embeddings = OpenAIEmbeddings()
db = Pinecone.from_existing_index(index_name="docchat",embedding=embeddings)

MODEL=os.environ["MODEL"]
if MODEL != 'GPT-4':
    QUESTION_LLM = ChatOpenAI(temperature=0, max_tokens=256)
    ANSWER_LLM = ChatOpenAI(temperature=0, max_tokens=768)
    MAX_CONTEXT_WORDS = 1150
else:
    QUESTION_LLM = ChatOpenAI(model_name='gpt-4',temperature=0, max_tokens=256,request_timeout=180)
    ANSWER_LLM = ChatOpenAI(model_name='gpt-4', temperature=0, max_tokens=1024,request_timeout=180)
    MAX_CONTEXT_WORDS = 1200

def ClearHistory(history):
    history.clear()
    return [], history

def user(message,bot_history, history):
     bot_history.append((message,None))
     return message,bot_history ,history

def expand(message,bot_history,history):
    question = GetQuestion(message,history)
    bot_history.append((question,None))
    return bot_history,question , history


def bot(message,bot_history,history):
    question = message
    response, full_response = GetAnswer(question,db)
    history.chat_memory.add_user_message(question)
    history.chat_memory.add_ai_message(response)
    bot_history.append((None,full_response))
    return bot_history,"" , history

with gr.Blocks() as demo:
    memory =gr.State(ConversationBufferWindowMemory(return_messages=True,memory_key="chat_history",k=4))
    gr.Markdown(f"<h1><center>DocBot Firmove.pl MODEL={MODEL}</center></h1>")
    chatbot = gr.Chatbot(label="Docbot v001").style(height=400)
    msg = gr.Textbox(label="Pytanie?")
    clear = gr.Button("Clear").style(full_width=False)

    msg.submit(user, [msg,chatbot, memory], [msg,chatbot, memory], queue=False).then(
        expand, [msg, chatbot, memory], [chatbot,msg ,memory]
    ).then (bot, [msg, chatbot, memory], [chatbot,msg ,memory])
    clear.click(ClearHistory, memory, [chatbot,memory], queue=False)

demo.launch()
