def get_history_user(db: Session, user: str):
    logger.info('get_history_user!')
    
    # Query the UserSession table filtered by the user and ordered by session_id and created_at
    sessions = db.query(UserSession).filter(UserSession.nbkid == user).order_by(UserSession.session_id, UserSession.created_at).all()
    
    # Group the sessions by session_id
    session_dict = defaultdict(list)
    for session in sessions:
        session_dict[session.session_id].append({
            'message_type': session.message_type,
            'message': session.message,
            'created_at': session.created_at
        })
    
    # Sort messages within each session by created_at in ascending order
    for session_id in session_dict:
        session_dict[session_id] = sorted(session_dict[session_id], key=lambda x: x['created_at'])
    
    # Format the output as a list of dictionaries with session_id as the key
    result = [
        {
            'session_id': session_id,
            'messages': session_dict[session_id]
        }
        for session_id in sorted(session_dict.keys(), key=lambda x: session_dict[x][-1]['created_at'], reverse=True)
    ]
    
    # Return the formatted chat history
    return result


def get_history_user(db: Session, user: str):
    logger.info('get_history_user!')
    
    # Query the UserSession table filtered by the user and ordered by session_id and created_at
    sessions = db.query(UserSession).filter(UserSession.nbkid == user).order_by(UserSession.session_id, UserSession.created_at).all()
    
    # Group the sessions by session_id
    session_dict = defaultdict(list)
    for session in sessions:
        session_dict[session.session_id].append(session)
    
    # Sort the session groups by their latest message timestamp in descending order
    sorted_sessions = sorted(session_dict.items(), key=lambda x: x[1][-1].created_at, reverse=True)
    
    # Return the grouped and sorted chat history
    return sorted_sessions
    
    
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc
from your_logging_module import logger  # Import your logger

# Assuming @log_performance is a custom decorator you've implemented
# @log_performance
def get_history_user(db: Session, user):
    logger.info('get_history_user!')
    
    # Query to get sessions of the user sorted by creation date (latest first)
    sessions = db.query(UserSession).filter(UserSession.nbkid == user).order_by(desc(UserSession.created_at)).all()
    
    # Create a dictionary to store the chat history grouped by session
    history = {}
    
    for session in sessions:
        # Query to get chats in each session sorted by creation date (ascending)
        chats = db.query(Chat).filter(Chat.session_id == session.id).order_by(asc(Chat.created_at)).all()
        history[session.id] = {
            'session_info': session,
            'chats': chats
        }
    
    return history







from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

def create_prompt():
    template = """You are a question answering Large Language Model.
    Answer the following question.
    USER: {question}
    ASSISTANT:"""
    prompt = PromptTemplate.from_template(template)
    return prompt

def create_llm_pipeline(model_id):
    llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    device=0,  # -1 for CPU
    batch_size=1,  
    model_kwargs={"do_sample": True, "max_length": 128},
    )
    return llm_pipeline

def create_chain(prompt, llm_pipeline):
    chain = prompt | llm_pipeline.bind(stop=["USER:"])
    return chain

if __name__ == "__main__":
    model_id = "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
    #model_id = "mistralai/Mistral-7B-v0.1"
    chain = create_chain(create_prompt(), create_llm_pipeline(model_id=model_id))

    keep_running = True
    while keep_running:
        print("input something...")
        question = input()
        if question != "exit":
            print(chain.invoke({"question": question}))
        else:
            keep_running = False
