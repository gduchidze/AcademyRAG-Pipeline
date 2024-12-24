import json
import operator
from json import JSONDecodeError
from typing import Optional, Callable, Any, Awaitable
from typing import TypedDict, Annotated, Sequence
from uuid import uuid4

import requests
from langchain_community.adapters.openai import convert_openai_messages
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.messages import (
    ToolMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph, add_messages
from openai import OpenAI
from pydantic import BaseModel, Field

from open_webui.utils.task import replace_prompt_variable, replace_messages_variable
from qdrant_client import QdrantClient

EmitterType = Optional[Callable[[dict], Awaitable[None]]]


def get_send_citation(__event_emitter__: EmitterType):
    async def send_citation(url: str, title: str, content: str):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": [content],
                    "metadata": [{"source": url, "html": False}],
                    "source": {"name": title},
                },
            }
        )

    return send_citation


def get_send_relevant_docs(__event_emitter__: EmitterType):
    async def send_relevant_docs(docs: list[Document]):
        if __event_emitter__ is None:
            return
        titles = [
            (
                doc.metadata["სათაური"]
                if "სათაური" in doc.metadata
                else doc.metadata["კატეგორია"]
            )
            for doc in docs
        ]

        merged_docs = {}
        for source, document in zip(titles, docs):
            if source not in merged_docs:
                merged_docs[source] = []
            merged_docs[source].append(document)

        for title, doc_group in merged_docs.items():
            await __event_emitter__(
                {
                    "type": "citation",
                    "data": {
                        "document": [doc_to_string(doc) for doc in doc_group],
                        "metadata": [{"source": title} for _ in doc_group],
                        "source": {"name": title},
                    },
                }
            )

    return send_relevant_docs


def get_send_status(__event_emitter__: EmitterType):
    async def send_status(
        status_message: str, done: bool = False, event_type: str = "status"
    ):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": event_type,
                "data": {"description": status_message, "done": done},
            }
        )

    return send_status


class Pipe:
    class Valves(BaseModel):
        PIPELINE_ID: str = Field(
            default="ailab-pipeline",
            description="Identifier for the pipeline model.",
        )
        PIPELINE_NAME: str = Field(
            default="AILab Pipeline", description="Name for the pipeline model."
        )
        ENABLE_EMITTERS: bool = Field(
            default=True,
            description="Toggle to enable or disable event emitters.",
        )
        OPENAI_BASE_URL: str = Field(
            default="https://api.ailab.ge",
            description="Base URL for OpenAI API endpoints",
        )
        OPENAI_API_KEY: str = Field(
            default="sk-MckIJPkrp42Ev4_EBkj6aQ",
            description="OpenAI API key",
        )
        MODEL_NAME: str = Field(
            default="kona-mini",
            # default="gpt-4o",
            description="Model Name",
        )

    def __init__(self):
        self.type = "manifold"
        self.__update_valves()
        print(f"{self.Valves.PIPELINE_ID}, __init__,  {self.valves=}")
        self.pipelines = self.pipes()

    def __update_valves(self):
        self.valves = self.Valves(
            **{k: v.default for k, v in self.Valves.model_fields.items()}
        )

    def pipes(self):
        return [{"name": self.valves.PIPELINE_NAME, "id": self.valves.PIPELINE_ID}]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __task__: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __tools__: Optional[dict] = None,
    ) -> str:
        self.__update_valves()
        print(f"{self.valves.PIPELINE_ID}, pipe,  {self.valves=}")
        print(f"{self.valves.PIPELINE_ID}, pipe,  {body=}")
        print(f"{self.valves.PIPELINE_ID}, pipe,  {__task__=}")
        print(f"{self.valves.PIPELINE_ID}, pipe,  {__user__=}")
        send_citation = get_send_citation(__event_emitter__)
        send_status = get_send_status(__event_emitter__)
        send_relevant_docs = get_send_relevant_docs(__event_emitter__)

        agent = AILabAgent(self.valves, send_status, send_citation, __user__)

        if __task__ == "title_generation":
            content = agent.llm.invoke(body["messages"]).content
            assert isinstance(content, str)
            yield content
            return

        inputs = {
            "query": body["messages"][-1]["content"],
            "messages": convert_openai_messages(body["messages"]),
        }

        async for event in agent.graph.astream_events(
            inputs, version="v2", stream_mode="values"
        ):
            kind = event["event"]
            data = event["data"]
            metadata = event.get("metadata", {})
            langgraph_node = metadata.get("langgraph_node")

            if kind == "on_chat_model_stream":
                if langgraph_node == "final_answer":
                    if "chunk" in data and (content := data["chunk"].content):
                        yield content
            elif kind == "on_chain_start":
                if langgraph_node == "tool_chooser":
                    await send_status(f"Routing the user query")
                elif langgraph_node == "query_rewriter":
                    await send_status(f"Rewriting the user query for knowledge search")
            elif kind == "on_tool_start":
                yield "\n"
                await send_status(f"Running tool {event['name']}", False)
            elif kind == "on_tool_end":
                await send_status(f"Tool '{event['name']}' finished", True)
                if event["name"] == "ailab_knowledge_search":
                    await send_relevant_docs(data.get("output"))
                    context = relevant_docs_to_context(data.get("output"))
                    if context.strip():
                        await send_citation(
                            url=event["name"],
                            title=event["name"],
                            content=f"გადაწერილი შეკითხვა: {data.get('input').get('query')}\n\n{context}",
                        )
                else:
                    await send_citation(
                        url=event["name"],
                        title=event["name"],
                        content=f"Tool '{event['name']}' with inputs {data.get('input')} returned:\n{data.get('output')}",
                    )


class AILabgentGraphState(TypedDict):
    query: str
    rewritten_query: Optional[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    relevant_docs: Annotated[Sequence[Document], operator.add]  # Change to Document
    intermediate_steps: Annotated[Sequence[BaseMessage], add_messages]


def embed_scores_into_docs(docs_with_scores):
    docs = []
    for doc, score in docs_with_scores:
        doc.metadata["score"] = round(score, 4)
        docs.append(doc)
    return docs


def convert_langchain_to_openai(messages):
    openai_format = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            openai_format.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            openai_format.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            openai_format.append({"role": "system", "content": msg.content})

    return openai_format


def doc_to_string(doc):
    metadata_keys = ["წყარო", "კატეგორია", "სათაური", "ქვესათაური", "score"]
    doc_content = ""
    for key in metadata_keys:
        if key in doc.metadata:
            if key == "ქვესათაური":
                section_names = doc.metadata[key].split("; ")
                if all(
                    [
                        f"# {section.strip()}" in doc.page_content
                        for section in section_names
                    ]
                ):
                    continue
            if key == "წყარო" and doc.metadata[key] in doc.page_content:
                continue
            if val := doc.metadata.get(key):
                doc_content += f"{key}: {val}\n"
    doc_content += f"\n{doc.page_content}"
    return doc_content


def relevant_docs_to_context(docs):
    contents = []
    for doc in docs:
        doc_content = doc_to_string(doc)
        contents.append(doc_content)
    return "\n------\n".join(contents)


class AILabAgent:
    def __init__(self, valves, send_status=None, send_citation=None, user=None):
        self.valves = valves
        self.ailab_knowledge_retriever = self.__get_knowledge_retriever()
        self.tools = self._get_ailab_tools()
        self.send_status = send_status
        self.send_citation = send_citation
        self.user = user
        lang_graph_request_id = str(uuid4())

        self.openai_client = OpenAI(
            base_url=self.valves.OPENAI_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
        )

        self.llm = ChatOpenAI(
            base_url=self.valves.OPENAI_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
            model_name=self.valves.MODEL_NAME,
            frequency_penalty=1.2,
            temperature=0.4,
            # extra_body={
            #     "metadata": {
            #         "call_type": "bog_final_answer",
            #         "lang_graph_request_id": lang_graph_request_id,
            #         "user": user,
            #     }
            # },
        )

        self.llm_query_rewriter = ChatOpenAI(
            base_url=self.valves.OPENAI_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
            model_name=self.valves.MODEL_NAME,
            frequency_penalty=1.2,
            temperature=0.4,
            # extra_body={
            #     "metadata": {
            #         "call_type": "bog_query_rewriter",
            #         "lang_graph_request_id": lang_graph_request_id,
            #         "user": user,
            #     }
            # },
        )

        self.llm_with_tools = ChatOpenAI(
            base_url=self.valves.OPENAI_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
            model_name=self.valves.MODEL_NAME,
            # frequency_penalty=1.2,
            temperature=0,
            max_tokens=64,
            disable_streaming="tool_calling",
            # extra_body={
            #     "metadata": {
            #         "call_type": "bog_tool_chooser",
            #         "lang_graph_request_id": lang_graph_request_id,
            #         "user": user,
            #     }
            # },
        ).bind_tools(
            [self.tools["bog_convert_currency"]],
            # tool_choice="bog_knowledge_search"
        )

        self.graph = self.__create_graph()

    def __get_knowledge_retriever(self):
        embeddings = DeepInfraEmbeddings(
            model_id="BAAI/bge-m3",
            deepinfra_api_token="sniRj9wr02CNUPLnq548PQAwnKNnFfK3",
        )
        qdrant_client = QdrantClient(
            url="https://0fe663a4-8b7e-4f5c-8a4d-b839e1a64d89.us-west-2-0.aws.cloud.qdrant.io",
            api_key="arnt83aOZ7wt6FIq2-SwsnEhCUbFjIZ0oGy0EuVEKH8KqMukH89MPw"
        )
        qdrant = Qdrant(
            client=qdrant_client,
            collection_name="ailab-notebooks",
            embeddings=embeddings,
        )
        return qdrant.as_retriever(
            search_kwargs={
                "k": 3,
                "score_threshold": 0.4
            }
        )

    def _get_ailab_tools(self):
        class KnowledgeSearchInput(BaseModel):
            query: str = Field(
                description="Verbose search query for AiLab's knowledge base."
            )

        @tool(args_schema=KnowledgeSearchInput)
        def ailab_knowledge_search(query: str):
            """
            Searches the query in AiLab's knowledge base. The search query must be provided
            in natural language and be verbose.
            """
            docs_with_scores = self.ailab_knowledge_retriever.vectorstore.similarity_search_with_relevance_scores(
                query, k=3, score_threshold=0.4
            )
            docs = embed_scores_into_docs(docs_with_scores)
            return docs

        return {
            "ailab_knowledge_search": ailab_knowledge_search,
        }

    def __create_graph(self):
        graph = StateGraph(AILabgentGraphState)
        graph.add_node("query_rewriter", lambda state: self.run_query_rewriter(state))
        graph.add_node("ailab_knowledge_search", lambda state: self.run_ailab_knowledge_search(state))
        graph.add_node("final_answer", lambda state: self.run_final_answer(state))

        graph.add_edge("query_rewriter", "ailab_knowledge_search")
        graph.add_edge("ailab_knowledge_search", "final_answer")
        graph.add_edge("final_answer", END)

        graph.set_entry_point("query_rewriter")

        return graph.compile()

    def run_ailab_knowledge_search(self, state):
        print(f"{self.valves.PIPELINE_ID}, run_ailab_knowledge_search , START")
        docs = self.tools["ailab_knowledge_search"].invoke(
            {"query": state["rewritten_query"]}
        )
        context = relevant_docs_to_context(docs)
        message = ToolMessage(content=context, tool_call_id="ailab_knowledge_search")
        print(f"{self.valves.PIPELINE_ID}, run_ailab_knowledge_search , END")
        return {"relevant_docs": [message], "intermediate_steps": [message]}

    def run_query_rewriter(self, state):
        print(f"{self.valves.PIPELINE_ID}, run_query_rewriter, START")

        query = state["query"]
        chat_history = [
            message
            for message in state["messages"]
            if not isinstance(message, SystemMessage)
        ]
        chat_history = convert_langchain_to_openai(chat_history)

        template = replace_prompt_variable(rewrite_template, query)
        message = replace_messages_variable(template, chat_history)
        messages = self.llm_query_rewriter.invoke(message)
        print(f"{self.valves.PIPELINE_ID}, run_query_rewriter rewritten_query={messages.content}, END")
        return {"rewritten_query": messages.content}

    def run_final_answer(self, state):
        print(f"{self.valves.PIPELINE_ID}, run_final_answer, START")
        query = state["query"]
        context = state["intermediate_steps"][-1].content
        if context:
            chat_history = build_rag_chat_history(context, query, state["messages"])
        else:
            chat_history = state["messages"]
        print(f"{self.valves.PIPELINE_ID}, run_final_answer , {chat_history=}")
        out = self.llm.invoke(chat_history)
        print(f"{self.valves.PIPELINE_ID}, run_final_answer, {out=}")
        print(f"{self.valves.PIPELINE_ID}, run_final_answer, END")
        return {"messages": [out]}

    def invoke(self, messages):
        return self.graph.invoke(
            {
                "query": messages[-1]["content"],
                "messages": convert_openai_messages(messages),
            }
        )


def insert_rag_template_as_user_message(context, query, messages):
    message_history = messages[:-1]
    message_history.append(
        HumanMessage(content=rag_template.format(context=context, query=query))
    )
    return message_history


def insert_rag_template_as_system_message(context, query, messages):
    system_messages = [
        message for message in messages if isinstance(message, SystemMessage)
    ]
    chat_history = [
        message for message in messages if not isinstance(message, SystemMessage)
    ]
    system_messages.append(
        SystemMessage(content=rag_template.format(context=context, query=query))
    )
    return system_messages + chat_history


def build_rag_chat_history(context, query, messages):
    system_message = SystemMessage(content=enhanced_rag_system_prompt)
    chat_history = [
        message for message in messages if not isinstance(message, SystemMessage)
    ]
    chat_history = convert_langchain_to_openai(chat_history)
    user_message_content = replace_messages_variable(rag_user_prompt, chat_history)
    user_message_content = user_message_content.format(context=context, query=query)
    user_message = HumanMessage(content=user_message_content)

    return [system_message, user_message]


rewrite_template = """მომხმარებლის შეკითხვისა და მიმოწერის გათვალისწინებით, გადააკეთე მისი შეკითხვა AI Lab - ის ხელოვნური ინტელექტის კურსის ინფორმაციის მოსაძებნად. შენ მიერ გადაკეთებული საძიებო ტექსტი უნდა იყოს ამომწურავი და შეიცავდეს მხოლოდ მომხმარებლის ბოლო მესიჯში ნაგულისხმევ ინფორმაციას. არ დაამატო სხვა ტექსტი, მხოლოდ დააბრუნე გადაწერილი query. გადაკეთებული საძიებო ტექსტი უნდა იყოს ქართულ ენაზე.
მიმოწერის ისტორია:
{{MESSAGES:END:5}}

მომხმარებლის მესიჯი:
{{prompt:end:500}}

საძიებო ტექსტი:
"""

rag_template = """მოცემულია AI Lab - ის ხელოვნური ინტელექტის კურსიდან მოძიებული დამატებითი ინფორმაცია, რომელიც <context> თეგებში ჩასმულია. უპასუხე შეკითხვას კონტექსტის მიხედვით და წესების გათვალისწინებით. 

<context>
{context}
</context>

დაიცავი შემდეგი წესები:
- უპასუხე მხოლოდ კონტექსტში მოცემული ინფორმაციის მიხედით.
- არ ჩასვა პასუხი html თეგებში.
- არ ახსენოთ კონტექსტის არსებობა შენს პასუხში.
- ნუ გამოიყენებთ კითხვა/პასუხის ფორმატს შენს პასუხში.

გაეცი პასუხი შემდეგ მოთხოვნას:
{query}

პასუხი:
"""

rag_system_prompt = """
შენ ხარ AI Lab - ის ხელოვნური ინტელექტის კურსის ასისტენტი. შენ მოგეწოდება მომხმარებლის და ასისტენტის მიმდინარე მიმოწერა და გაქვს კურსის მასალებზე წვდომა რომელის მიხედვითაც მომხმარებლელს კურსის შესახებ ბოლო შეკითხვაზე/მოთხოვნაზე გასცემ პასუხს. ეს დამხმარე ინფორმაცია <context> თეგებში იქნება მოცემული.
უპასუხე მომხმარებლის ბოლო შეკითხვას უკვე არსებული მიმოწერის მიხედვით, მოძიებულ ინფორმაციაზე დაყრდნობით და წესების გათვალისწინებით.

დაიცავი შემდეგი წესები:
- უპასუხე მხოლოდ კონტექსტში მოცემული ინფორმაციის მიხედით.
- არ ჩასვა პასუხი html თეგებში.
- არ ახსენოთ კონტექსტის არსებობა შენს პასუხში.
- ნუ გამოიყენებთ კითხვა/პასუხის ფორმატს შენს პასუხში.
"""

enhanced_rag_system_prompt = """შენ ხარ AI Lab - ის ხელოვნური ინტელექტის ასისტენტი. შენი მთავარი მიზანია მომხმარებლებს მიაწოდო სასარგებლო და ზუსტი ინფორმაცია მასალებიდან აუხსნა ის დეტალურად მარტივი მაგალითებით როგორც ამას აუხსნიდი მოზარდს. შენ მოგეწოდება მომხმარებლის და ასისტენტის მიმდინარე მიმოწერა და AI Lab - ის ხელოვნური ინტელექტის კურსის მასალები და მაქედან მოძიებული ინფორმაცია, რომელიც მომხმარებლის ბოლო შეკითხვაზე/მოთხოვნახე დაგეხმარება. ეს დამხმარე ინფორმაცია <context> თეგებში იქნება მოცემული.
უპასუხე მომხმარებლის ბოლო შეკითხვას უკვე არსებული მიმოწერის მიხედვით, მოძიებულ ინფორმაციაზე დაყრდნობით და წესების გათვალისწინებით.

context-ის გამოყენების წესები:
- უპასუხე მხოლოდ კონტექსტში მოცემული ინფორმაციის მიხედით.
- არ ახსენოთ კონტექსტის არსებობა შენს პასუხში.
- ნუ გამოიყენებთ კითხვა/პასუხის ფორმატს შენს პასუხში.
- პასუხში გამოიყენე რელევანტური ინფორმაციები მასალებიდან.

მომხმარებელთან კომუნიკაციის წესები:
- არავითარ შემთხვევაში არ უპასუხო შეკითხვებს და მოთხოვნებს, რომელიც არ შეეხება AI Lab ის ხელოვნური ინტელექტის კურსს.
- მომხარებელს დაუბრუნე ზუსტი, დეტალური და ამომწურავი პასუხი, გამოიყენე bullet point-ები და **bold** ფორმატირება.
- თუ მომხმარებლის შეკითხვა არ არის კონკრეტული, მომხმარებელს სთხოვე დააკონკრეტოს კონკრეტულად რა აინტერესებს. 
"""

rag_user_prompt = """მიმდინარე მიმოწერა მომხმარებელსა და AI Lab - ის ციფრულ ასისტენტს შორის:
{{MESSAGES:END:5}}

<context>
{context}
</context>

უპასუხე მომხმარებლის ბოლო მესიჯს:
{query}"""
