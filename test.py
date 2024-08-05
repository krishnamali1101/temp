
from transformers import AutoModelForCausalLM, AutoTokenizer

def create_chatbot(model_name):
  """Creates a chatbot using a Hugging Face LLM."""
  model = AutoModelForCausalLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  def chatbot(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)  # Adjust max_length as needed
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

  return chatbot



import mistral

def create_chatbot():
  """Creates a chatbot using the Mistral language model."""
  llm = mistral.load("mistral-7b-instruct")  # Replace with your desired Mistral model

  def chatbot(user_input):
    """Handles user input and generates a response."""
    response = llm.generate(user_input, max_tokens=100)  # Adjust max_tokens as needed
    return response.text

  return chatbot





Updated Database Schema
Tables
Users

user_id: Primary Key, unique identifier for each user.
username: User's name.
email: User's email address.
password: User's hashed password.
Buckets

bucket_id: Primary Key, unique identifier for each bucket or sub-bucket.
bucket_name: Name of the bucket or sub-bucket (e.g., userid/private_buckets/bkt1, sub_bucket).
parent_bucket_id: Foreign Key, references bucket_id in the same table, for linking sub-buckets to their parent buckets (NULL for top-level buckets).
owner_id: Foreign Key, references user_id in the Users table, to store the owner/creator of the bucket.
AccessControl

access_id: Primary Key, unique identifier for each access control entry.
user_id: Foreign Key, references user_id in the Users table.
bucket_id: Foreign Key, references bucket_id in the Buckets table.
access_level: Type of access granted (e.g., read, write, admin).


from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    password = Column(String(255), nullable=False)
    
    owned_buckets = relationship("Bucket", back_populates="owner")
    access_controls = relationship("AccessControl", back_populates="user")

class Bucket(Base):
    __tablename__ = 'buckets'
    
    bucket_id = Column(Integer, primary_key=True, autoincrement=True)
    bucket_name = Column(String(255), nullable=False)
    parent_bucket_id = Column(Integer, ForeignKey('buckets.bucket_id'), nullable=True)
    owner_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    
    parent_bucket = relationship("Bucket", remote_side=[bucket_id])
    sub_buckets = relationship("Bucket", back_populates="parent_bucket", remote_side=[parent_bucket_id])
    owner = relationship("User", back_populates="owned_buckets")
    access_controls = relationship("AccessControl", back_populates="bucket")

class AccessControl(Base):
    __tablename__ = 'access_control'
    
    access_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    bucket_id = Column(Integer, ForeignKey('buckets.bucket_id'), nullable=False)
    access_level = Column(Enum('read', 'write', 'admin', name='access_level_enum'), nullable=False)
    
    user = relationship("User", back_populates="access_controls")
    bucket = relationship("Bucket", back_populates="access_controls")

# Database connection setup
DATABASE_URL = "mysql+pymysql://user:password@localhost/dbname"  # Replace with your actual database URL

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Create tables
Base.metadata.create_all(engine)










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
