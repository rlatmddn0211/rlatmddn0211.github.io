---
layout: post
title: "LangChain & RAG [Stuff/Map Reduce Chain Implementation] "
date: 2026-01-13 14:00:00 +0900
categories: LangChain, LLM, Prompting,Memory
tags: [
    LLM, Prompt_Template, Prompting, FewShotPrompt,StuffChain,MapReduceChain
]
---


# RAG_Implementation_LCEL (Stuff / Map Reduce)

# Stuff Chain Implementation LCEL

## Stuff Chain

- 주어진 문단을 모델에게 통째로 전달하여 주어진 문서에 대한 답변을 생성

### Chain의 구성요소

![](/assets/langchain/rag_implementation/1.png)

- **Retriever :**
    - **input : 질문과 관련이 있는 Document를 얻기 위한 query**
    - **output :  Document들의 list (질문과 관련이 있는)**
- 우선 LCEL을 통해 stuff chain을 구현하기 위해서는 retriever와 prompt를 생성해줘야한다.

<aside>

### Stuff Chain Implementation

- 질문을 query하여 관련이 있는 지문을 먼저 찾고, 찾은 지문 전체를 prompt로 활용하여 LLM에게 전달
- retriever, prompt, llm (chain 구성)
</aside>

**retriever**

```python
retriever=vectorstore.as_retriever()
```

**Prompt**

```python
from langchain.prompts import ChatPromptTemplate
prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. 
    Answer questions using only the following context. 
    If you don't know the answer just say you don't know, 
    don't make it up : \n{context}"),
    ("human","{question}"),
])
```

Chain

💡 **Chain 내에서 retriever는 자동으로 호출된다. (input은 사용자의 질문임)** 

```python
chain=retriever|prompt|llm
chain.invoke({"Describe Victory Mansions"})
```

→ 하지만, 이렇게만 구현하면 작동하지 않는다.

- **우선, 이렇게 실행하면, 주어진 string (Describe .. )는 retriever에게 주어질 것임.**
- **이후에, retriever가 선택한 documents를 list로 반환하며, 그 list는 {context}에 저장될 것임**
- **또한, 주어진 질문은 question에도 들어가야함.**

**→ 이러한 구체적인 동작들을 구현해줘야함**

1. **Retriever가 질문과 관련된 문서를 선택 → prompt의 context로 전달**
    - chain을 구성할때에, “context”라는 변수명에, retriever의 출력값이 담겨, 다음 chain으로 넘어감
2. **주어진 question은 retriever 뿐만 아니라, prompt에도 전달 되어야함.**
    - RunnablePassThrough() 는 chain의 다음 구성요소로 넘기는 역할을 함 (입력값을 넘김)

```python
from langchain.schema.runnable import RunnablePassthrough

retriever=vectorstore.as_retriever()
prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up : \n{context}"),
    ("human","{question}"),
])

chain=({
    "context":retriever,
    "question":RunnablePassthrough(),
    }
    |prompt
    |llm
)

chain.invoke({"Describe Victory Mansions"})

>>>
AIMessage(content='Victory Mansions is a building in London that is described as having a gritty and unpleasant environment. The hallway smells of boiled cabbage and old rag mats. The building is seven flights up, with a broken lift that is rarely working. There is a large colored poster of a man\'s face with a caption that reads "BIG BROTHER IS WATCHING YOU" on each landing. The building is one of four similar structures in London that house the Ministries of Truth, Peace, Love, and Plenty.')
```

- 흐름을 정리해보면, “Describe Victory Mansions”는 Retriever의 input으로 들어감
    - retriever는 input으로 single string을 입력받음
- 이후, 그 질문과 관련된 문서들을 context라는 변수에 담아 chain의 다음 구성요소에 전달
    - retriever의 output은 질문과 관련된 Documents들의 list임

![](/assets/langchain/rag_implementation/2.png)

- LangSmith를 통해 구현한 동작을 확인할 수 있음
    - Retriever는 주어진 문서의 문단 중, 질문과 관련이 있는 text 4개를 반환했음 (context라는 변수에 담아서)

![](/assets/langchain/rag_implementation/3.png)

- Prompt는 retriever가 찾은 4개의 Documents를 받음
    
    ![](/assets/langchain/rag_implementation/4.png)
    
- LLM은 input으로 system에게 전달될 prompt를 받음

---

# Map Reduce Chain Implementation LCEL

### Map Reduce Chain

- Retriever가 선택한 question과 관련된 documents들을 모두 요약하여 LLM에게 전달

<aside>

### Map Reduce Chain Implementation

1. List of Documents (Retriever가 질문과 관련된 Documents들을 선택)
2. List of Documents안에 있는 문서 하나하나 별로 질문에 대한 답변을 생성 (문서 개수만큼 답변이 생성됨)
3. 저장된 llm의 결과를 prompt로 합쳐 llm이 최종 결과를 만들어냄
</aside>

```python
llm=ChatOpenAI(
    temperature=0.1
)
loader=UnstructuredFileLoader("../files/chapter_one.docx")
splitter=CharacterTextSplitter(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)
docs=loader.load_and_split(text_splitter=splitter)
embeddings=OpenAIEmbeddings()
cache_dir=LocalFileStore("../.cache/")
cached_embeddings=CacheBackedEmbeddings.from_bytes_store(embeddings,cache_dir)
vectorstore=FAISS.from_documents(docs,cached_embeddings)
retriever=vectorstore.as_retriever()
```

- 기존과 동일, 문서들을 load, split, caching, vectorstore, retriever 선언

```python
map_doc_prompt=ChatPromptTemplate.from_messages([
    ("system",
     """Use the following portion of a long document to see if any of the text 
     is relevant to answer the question. Return any relevant text verbatim.
     -------
     {context}
     """,
    ),
    ("human","{question}"),
])
map_doc_chain=map_doc_prompt|llm
def map_docs(inputs):
    documents=inputs['documents']
    question=inputs['question']
    return "\n\n".join(
        map_doc_chain.invoke(
            {
                "context":doc.page_content,
                "question":question,
            }
        ).content 
        for doc in documents
    )
    

#extract list of docs through retriever
#create response about each docs
#merge all the responses and use as prompt for llm
map_chain=({
    "documents":retriever,
    "question":RunnablePassthrough()
    }
    |RunnableLambda(map_docs)
)
#map_chain의 output (이런 형식으로 출력될 것임)
#{
#     "documents":[Documents],
#     "question":"Describe Victory Mansions"
#}
final_prompt=ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful assistant. Answer questions using only the 
     following context. If you don't know the answer just say you don't know, 
     Don't try to make up an answer.
     -------
     {context}
     """,
    ),
    ("human","{question}"),
])
chain=(
    {"context": map_chain,
    "question":RunnablePassthrough(),
    }
    |final_prompt|llm
)
chain.invoke("Where does Winston go to work?")
```

- Map Reduce이기 때문에, retriever가 선택한 documents 각각에 대해서 LLM은 response를 생성하고, 그 각각의 response를 취합하여 최종적으로 LLM에게 전달하여 최종 답변을 얻는다.

우선 코드는 크게 2개의 부분으로 나눌 수 있다. 

1. retriever가 선택한 documents들 각각에 대한 LLM response 생성
    - retriever의 documents를 하나의 list에 저장, llm에 전달하여 각각에 대한 response 생성
    - map_doc_chain(map_doc_prompt | llm)
        - map_doc_prompt : retriever가 추출한 document 각각에 대해 답변을 생성하도록 할 prompt
            - context : documents extracted by retreiver
            - question : user’s question
    - map_docs()
        - 취합된 documents들을 하나하나씩 for문으로 돌며 각 documents에 대한 response 생성
    - map_chain()
        - map_doc_chain과 chain을 이어주는 역할
        - map_doc_chain에서 생성된 각 documents들에 대한 response를 취합하여 chain의 llm에게 전달
2. 합친 response를 최종적으로 LLM에게 전달
    - chain(”context”,”question)
        - 실질적으로 1번에서 취합한 response에 대한 chain