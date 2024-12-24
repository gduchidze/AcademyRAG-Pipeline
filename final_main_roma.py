import operator
from typing import Optional, Callable, Any, Awaitable, List
from typing import TypedDict, Annotated, Sequence

from langchain_community.adapters.openai import convert_openai_messages
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
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from open_webui.utils.task import replace_prompt_variable, replace_messages_variable




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
    async def send_relevant_docs(docs: list[str]):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": docs,
                    "metadata": [{"source": "AILab Knowledge Base"} for _ in docs],
                    "source": {"name": "AILab Search Results"},
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
            default="AI Lab Assistant", description="Name for the pipeline model."
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
            default="tbilisi-ai-lab-2.0",
            # default="gpt-4o",
            description="Model Name",
        )

    def __init__(self):
        self.type = "manifold"
        self.__update_valves()
        print(f"{self.valves=}")
        self.pipelines = self.pipes()

    def __update_valves(self):
        self.valves = self.Valves(
            **{k: v.default for k, v in self.Valves.model_fields.items()}
            # **{k: os.getenv(k, v.default) for k, v in self.Valves.model_fields.items()}
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
        print(f"{self.valves=}")
        print(f"{body=}")
        print(f"{__task__=}")
        print(f"{__user__=}")
        send_citation = get_send_citation(__event_emitter__)
        send_status = get_send_status(__event_emitter__)

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


class AILabAgentGraphState(TypedDict):
    query: str
    rewritten_query: Optional[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    relevant_docs: Annotated[Sequence[Document], operator.add]
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


def relevant_docs_to_context(docs: list[str]) -> str:
    return "\n------\n".join(docs)


class CustomEmbeddings(Embeddings):
    def __init__(self):
        self.client = OpenAI(
            api_key="sniRj9wr02CNUPLnq548PQAwnKNnFfK3",
            base_url="https://api.deepinfra.com/v1/openai"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = self.client.embeddings.create(
                model="BAAI/bge-m3",
                input=text,
                encoding_format="float"
            ).data[0].embedding
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class AILabAgent:
    def __init__(self, valves, send_status=None, send_citation=None, user=None):
        self.valves = valves
        self.send_status = send_status
        self.send_citation = send_citation
        self.user = user

        self.openai_client = OpenAI(
            base_url=self.valves.OPENAI_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
        )

        self.embeddings = CustomEmbeddings()
        self.vector_store = QdrantVectorStore(
            collection_name="ai-lab-academy-courses",
            embedding=self.embeddings,
            url="https://6e283805-9178-46dd-86bf-285943e9ffab.eu-west-1-0.aws.cloud.qdrant.io:6333",
            prefer_grpc=True,
            api_key="arnt83aOZ7wt6FIq2-SwsnEhCUbFjIZ0oGy0EuVEKH8KqMukH89MPw",
        )

        self.deepinfra_client = OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key="sniRj9wr02CNUPLnq548PQAwnKNnFfK3",
        )

        self.llm = ChatOpenAI(
            base_url=self.valves.OPENAI_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
            model_name=self.valves.MODEL_NAME,
            frequency_penalty=1.2,
            temperature=0.4,
            streaming=False,
        )

        self.llm_query_rewriter = ChatOpenAI(
            base_url=self.valves.OPENAI_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
            model_name=self.valves.MODEL_NAME,
            frequency_penalty=1.2,
            temperature=0.4,
        )

        self.tools = self._get_ailab_tools()
        self.graph = self.__create_graph()

    def _get_ailab_tools(self):
        class KnowledgeSearchInput(BaseModel):
            query: str = Field(
                description="Verbose search query for AiLab's knowledge base."
            )

        @tool(args_schema=KnowledgeSearchInput)
        def ailab_knowledge_search(query: str):
            try:
                print(f"Searching with query: {query}")
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=3,
                    score_threshold=0.3
                )

                if not results:
                    return ["ინფორმაცია ვერ მოიძებნა."]

                formatted_results = []
                used_files = set()

                for doc, score in results:
                    file_name = doc.metadata.get("file", "")
                    if file_name in used_files:
                        continue

                    used_files.add(file_name)
                    content = doc.page_content
                    content = content.replace("Markdown:", "").replace("<br/>", "\n").strip()
                    formatted_results.append(content)

                return (
                    formatted_results
                    if formatted_results
                    else ["ინფორმაცია ვერ მოიძებნა."]
                )

            except Exception as e:
                print(f"Error in ailab_knowledge_search: {str(e)}")
                return ["მოხდა შეცდომა ძიების პროცესში."]

        return {
            "ailab_knowledge_search": ailab_knowledge_search,
        }

    def __create_graph(self):
        graph = StateGraph(AILabAgentGraphState)
        graph.add_node("query_rewriter", lambda state: self.run_query_rewriter(state))
        graph.add_node(
            "ailab_knowledge_search",
            lambda state: self.run_ailab_knowledge_search(state),
        )
        graph.add_node("final_answer", lambda state: self.run_final_answer(state))

        graph.add_edge("query_rewriter", "ailab_knowledge_search")
        graph.add_edge("ailab_knowledge_search", "final_answer")
        graph.add_edge("final_answer", END)

        graph.set_entry_point("query_rewriter")

        return graph.compile()

    def run_ailab_knowledge_search(self, state):
        print("> ailab_knowledge_search")
        try:
            if "rewritten_query" not in state or not state["rewritten_query"]:
                print("No rewritten query found in state")
                return {
                    "relevant_docs": [],
                    "intermediate_steps": [
                        ToolMessage(
                            content="შეკითხვა ვერ მოიძებნა",
                            tool_call_id="ailab_knowledge_search",
                        )
                    ],
                }

            query = state["rewritten_query"]
            print(f"Searching for query: {query}")

            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=3,
                score_threshold=0.3
            )

            formatted_results = []
            used_files = set()

            for doc, score in results:
                file_name = doc.metadata.get("file", "")
                if file_name in used_files:
                    continue

                used_files.add(file_name)
                content = doc.page_content
                content = content.replace("Markdown:", "").replace("<br/>", "\n").strip()
                formatted_results.append(content)

            if not formatted_results:
                return {
                    "relevant_docs": [],
                    "intermediate_steps": [
                        ToolMessage(
                            content="ინფორმაცია ვერ მოიძებნა",
                            tool_call_id="ailab_knowledge_search",
                        )
                    ],
                }

            context = "\n---\n".join(formatted_results)
            message = ToolMessage(content=context, tool_call_id="ailab_knowledge_search")

            return {
                "relevant_docs": formatted_results,
                "intermediate_steps": [message]
            }

        except Exception as e:
            print(f"Error in run_ailab_knowledge_search: {str(e)}")
            return {
                "relevant_docs": [],
                "intermediate_steps": [
                    ToolMessage(
                        content="სისტემური შეცდომა",
                        tool_call_id="ailab_knowledge_search",
                    )
                ]
            }

    def run_query_rewriter(self, state):
        print("> run_query_rewriter")
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
        print(f"rewritten_query={messages.content}")
        return {"rewritten_query": messages.content}

    def run_final_answer(self, state):
        print("> run_final_answer")
        query = state["query"]
        context = state["intermediate_steps"][-1].content
        if context:
            print("> if context")
            chat_history = build_rag_chat_history(context, query, state["messages"])
        else:
            chat_history = state["messages"]
        print(f"{chat_history=}")
        out = self.llm.invoke(chat_history)
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
    try:
        print("> build_rag_chat_history start")
        system_message = SystemMessage(content=enhanced_rag_system_prompt)
        chat_history = [
            message for message in messages if not isinstance(message, SystemMessage)
        ]
        chat_history = convert_langchain_to_openai(chat_history)
        user_message_content = replace_messages_variable(rag_user_prompt, chat_history)
        user_message_content = user_message_content.format(context=context, query=query)
        user_message = HumanMessage(content=user_message_content)
        print("> build_rag_chat_history end")
        return [system_message, user_message]
    except Exception as e:
        print(f"Error in build_rag_chat_history: {str(e)}")
        return messages


rewrite_template = """მომხმარებლის შეკითხვისა და მიმოწერის გათვალისწინებით, გადააკეთე მისი შეკითხვა AI Lab - ის ხელოვნური ინტელექტის კურსის ინფორმაციის მოსაძებნად. შენ მიერ გადაკეთებული საძიებო ტექსტი უნდა იყოს ამომწურავი მაგრამ ლაკონური და შეიცავდეს მხოლოდ მომხმარებლის ბოლო მესიჯში ნაგულისხმევ ინფორმაციას. არ დაამატო სხვა ტექსტი, მხოლოდ დააბრუნე გადაწერილი query. გადაკეთებული საძიებო ტექსტი უნდა იყოს ქართულ ენაზე.
მაგალითები:
"ვერ ვსწავლობ მიჭირს ნეირონული ქსელების გაგება და შეგიძლია ამიხსნა" - "ნეირონული ქსელების ახსნა"

"შეგიძლია ამიხსნა წრფივი რეგრესია და ლოჯისტიკური რეგრესია რითი განსხვავდება ერთმანეთისგან"

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
- მომხარებელს დაუბრუნე ზუსტი, დეტალური და ამომწურავი პასუხი, არ გამოიყენო bullet point-ები და **bold** ფორმატირება.
- თუ მომხმარებლის შეკითხვა არ არის კონკრეტული, მომხმარებელს სთხოვე დააკონკრეტოს კონკრეტულად რა აინტერესებს. 
"""

rag_user_prompt = """მიმდინარე მიმოწერა მომხმარებელსა და AI Lab - ის ციფრულ ასისტენტს შორის:
{{MESSAGES:END:5}}

<context>
{context}
</context>

უპასუხე მომხმარებლის ბოლო მესიჯს:
{query}"""
