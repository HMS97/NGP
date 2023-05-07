from fastapi import FastAPI
from typing import List, Dict,Union
from video import VideoSeparator
from tool import *
import numpy as np
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
from langchain.document_loaders import TextLoader
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os 
from langchain.callbacks import get_openai_callback
from tqdm import tqdm
os.environ["OPENAI_API_KEY"] = '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'

from langchain.document_loaders import DirectoryLoader
from tool import *
from QA_Frame import init_db, question2interval
# from fastapi_cache import FastAPICache
# from fastapi_cache.backends.redis import RedisBackend
# from fastapi_cache.decorator import cache

# from redis import asyncio as aioredis

from cashews import cache
cache.setup("mem://")  

def face_data(vs,topK=None):
    
    formatted_intervals = lambda intervals :[{"start": start, "end": end} for start, end in intervals]

    padding_video_dict = pad_video_dict(vs.pack_data['face_records'], padding =5)
    intervals_padding_video_dict = get_continuing_intervals(padding_video_dict)

    length_list = [len(vs.pack_data["face_records"][key]) for key in vs.pack_data["face_records"].keys()]

    print(vs.pack_data["face_records"].keys(), vs.pack_data["face_images"].keys())
    #top k face images index K 
    if topK == None:
        top_index = np.argsort(length_list)[-10:]
    else:
        top_index = np.argsort(length_list)[-topK:]

    top_index = top_index[::-1]
    face_list = list(vs.pack_data["face_records"].keys())
    top_index_image = []
    for index in top_index:
        face_images = vs.pack_data['face_images'][face_list[index]]
        top_index_image.append((face_images[len(face_images)//2]+1)*255/2)

    top_index_intervals = []
    for index in top_index:
        top_index_intervals.append(formatted_intervals(intervals_padding_video_dict[face_list[index]]))


    description_list = []
    for index ,(ID,IMAGE,INTERVAL) in enumerate(zip(top_index, top_index_image,top_index_intervals)):
        description_list.append({"ID":int(ID),"face":convert_image(IMAGE),"intervals":INTERVAL})

    return description_list





vs = VideoSeparator(load_data_path="data.pkl")
dataset_db,dataset_text_list= init_db(data_dir="data")

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/video-data/{top_k}")
@cache(ttl="3h")
async def get_video_data(top_k: int = None) -> List[Dict[str, Union[int, bytes, List[Dict[str, int]]]]]:

    description_list = face_data(vs, top_k)
    response_bytes = bytes(json.dumps(description_list), encoding="utf-8")

    # print(description_list)
    return StreamingResponse(io.BytesIO(response_bytes), media_type="application/json")





@app.get("/ask_question/{question}")
@cache(ttl="3h")
async def ask(question: str = None) -> List[Dict[str, Union[int, bytes, List[Dict[str, int]]]]]:
    print(question)
    interval_result = question2interval(question,dataset_db,dataset_text_list)

    response_bytes = bytes(json.dumps(interval_result), encoding="utf-8")

    return StreamingResponse(io.BytesIO(response_bytes), media_type="application/json")



