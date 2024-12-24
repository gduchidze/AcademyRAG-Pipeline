import operator
import re
from typing import Optional, Callable, Any, Awaitable
from typing import TypedDict, Annotated, Sequence

from langchain_community.adapters.openai import convert_openai_messages
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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.constants import END
from langgraph.graph import StateGraph, add_messages
from openai import OpenAI
from pydantic import BaseModel, Field
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
                    search_results = data.get("output", [])
                    await send_relevant_docs(search_results)
                    context = relevant_docs_to_context(search_results)
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


class AILabAgentGraphState(TypedDict):
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


def relevant_docs_to_context(docs: list[str]) -> str:
    return "\n------\n".join(docs)


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

        self.qdrant_client = QdrantClient(
            url="https://0fe663a4-8b7e-4f5c-8a4d-b839e1a64d89.us-west-2-0.aws.cloud.qdrant.io",
            api_key="arnt83aOZ7wt6FIq2-SwsnEhCUbFjIZ0oGy0EuVEKH8KqMukH89MPw"
        )

        self.llm = ChatOpenAI(
            base_url=self.valves.OPENAI_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
            model_name=self.valves.MODEL_NAME,
            frequency_penalty=1.2,
            temperature=0.4,
            streaming=False
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

    def __get_knowledge_retriever(self):

        embeddings = OpenAIEmbeddings(
            openai_api_key="hHnQ6vbPMRKj7eCs5IG6QmjFTyYjVccW",
            openai_api_base="https://api.deepinfra.com/v1/openai",
            model="BAAI/bge-m3"
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
            Searches the query in AiLab's knowledge base using direct Qdrant search.
            """
            try:
                print(f"Searching with query: {query}")

                embedding_client = OpenAI(
                    base_url="https://api.deepinfra.com/v1/openai",
                    api_key="sniRj9wr02CNUPLnq548PQAwnKNnFfK3"
                )
                query_embedding = embedding_client.embeddings.create(
                    model="BAAI/bge-m3",
                    input=query,
                    encoding_format="float"
                ).data[0].embedding
                results = self.qdrant_client.search(
                    collection_name="ailab-notebooks",
                    query_vector=query_embedding,
                    limit=3,
                    score_threshold=0.4
                )

                if not results:
                    return ["ინფორმაცია ვერ მოიძებნა."]

                formatted_results = []
                for point in results:
                    if point.payload and 'content' in point.payload:
                        formatted_results.append(point.payload['content'])

                return formatted_results if formatted_results else ["ინფორმაცია ვერ მოიძებნა."]

            except Exception as e:
                print(f"Error in ailab_knowledge_search: {str(e)}")
                return ["მოხდა შეცდომა ძიების პროცესში."]

        return {
            "ailab_knowledge_search": ailab_knowledge_search,
        }

    def __create_graph(self):
        graph = StateGraph(AILabAgentGraphState)
        graph.add_node("query_rewriter", lambda state: self.run_query_rewriter(state))
        graph.add_node("ailab_knowledge_search", lambda state: self.run_ailab_knowledge_search(state))
        graph.add_node("final_answer", lambda state: self.run_final_answer(state))

        graph.add_edge("query_rewriter", "ailab_knowledge_search")
        graph.add_edge("ailab_knowledge_search", "final_answer")
        graph.add_edge("final_answer", END)

        graph.set_entry_point("query_rewriter")

        return graph.compile()

    def run_ailab_knowledge_search(self, state):
        """
        გაუმჯობესებული ძიების ფუნქცია კონტექსტის უკეთესი დამუშავებით.
        """
        print("> ailab_knowledge_search")
        try:
            if "rewritten_query" not in state or not state["rewritten_query"]:
                print("No rewritten query found in state")
                return {
                    "relevant_docs": [],
                    "intermediate_steps": [ToolMessage(
                        content="შეკითხვა ვერ მოიძებნა",
                        tool_call_id="ailab_knowledge_search"
                    )]
                }

            query = state["rewritten_query"]
            print(f"Searching for query: {query}")

            try:
                embedding_client = OpenAI(
                    base_url="https://api.deepinfra.com/v1/openai",
                    api_key="sniRj9wr02CNUPLnq548PQAwnKNnFfK3"
                )

                query_embedding = embedding_client.embeddings.create(
                    model="BAAI/bge-m3",
                    input=query,
                    encoding_format="float"
                ).data[0].embedding

                # დავამატოთ timeout Qdrant ძიებისთვის
                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self.qdrant_client.search,
                        collection_name="ailab-notebooks",
                        query_vector=query_embedding,
                        limit=5,
                        score_threshold=0.3,
                        with_payload=True
                    )
                    try:
                        results = future.result(timeout=5)  # 5 წამიანი timeout
                    except TimeoutError:
                        return {
                            "relevant_docs": [],
                            "intermediate_steps": [ToolMessage(
                                content="ქდრანტთან დაკავშირება ვერ მოხერხდა 5 წამის განმავლობაში",
                                tool_call_id="ailab_knowledge_search"
                            )]
                        }

                formatted_results = []
                used_files = set()

                for result in results:
                    if not result.payload or 'text' not in result.payload:
                        continue

                    file_name = result.payload.get('file', '')
                    if file_name in used_files:
                        continue

                    used_files.add(file_name)
                    content = result.payload['text']

                    content = content.replace('Markdown:', '').replace('<br/>', '\n')
                    content = re.sub(r'#*\s*', '', content)

                    formatted_results.append(content)

                if not formatted_results:
                    return {
                        "relevant_docs": [],
                        "intermediate_steps": [ToolMessage(
                            content="ინფორმაცია ვერ მოიძებნა",
                            tool_call_id="ailab_knowledge_search"
                        )]
                    }

                context = "\n---\n".join(formatted_results)
                message = ToolMessage(content=context, tool_call_id="ailab_knowledge_search")

                return {
                    "relevant_docs": formatted_results,
                    "intermediate_steps": [message]
                }

            except Exception as e:
                print(f"Error during knowledge search: {str(e)}")
                return {
                    "relevant_docs": [],
                    "intermediate_steps": [ToolMessage(
                        content="შეცდომა მოხდა ძიებისას",
                        tool_call_id="ailab_knowledge_search"
                    )]
                }

        except Exception as e:
            print(f"Error in run_ailab_knowledge_search: {str(e)}")
            return {
                "relevant_docs": [],
                "intermediate_steps": [ToolMessage(
                    content="სისტემური შეცდომა",
                    tool_call_id="ailab_knowledge_search"
                )]
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
    # return chat_history
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


def replace_prompt_variable(template: str, prompt: str) -> str:
    """
    Replaces {{prompt:end:N}} in template with the prompt
    """
    import re

    # Find all prompt variables in the template
    prompt_vars = re.findall(r'\{\{prompt:end:\d+\}\}', template)

    result = template
    for var in prompt_vars:
        # Extract limit from the variable
        limit = int(re.search(r':(\d+)\}\}', var).group(1))
        # Replace variable with truncated prompt
        result = result.replace(var, prompt[:limit])

    return result


def replace_messages_variable(template: str, messages: list) -> str:
    """
    Replaces {{MESSAGES:END:N}} in template with the last N messages
    """
    import re
    message_vars = re.findall(r'\{\{MESSAGES:END:\d+\}\}', template)

    result = template
    for var in message_vars:
        n = int(re.search(r':(\d+)\}\}', var).group(1))
        formatted_messages = []
        for msg in messages[-n:]:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted_messages.append(f"{role}: {content}")
        result = result.replace(var, '\n'.join(formatted_messages))

    return result



rewrite_template = """მომხმარებლის შეკითხვისა და მიმოწერის გათვალისწინებით, გადააკეთე მისი შეკითხვა AI Lab - ის ხელოვნური ინტელექტის კურსის ინფორმაციის მოსაძებნად. შენ მიერ გადაკეთებული საძიებო ტექსტი უნდა იყოს ამომწურავი მაგრამ ლაკონური და შეიცავდეს მხოლოდ მომხმარებლის ბოლო მესიჯში ნაგულისხმევ ინფორმაციას. არ დაამატო სხვა ტექსტი, მხოლოდ დააბრუნე გადაწერილი query. გადაკეთებული საძიებო ტექსტი უნდა იყოს ქართულ ენაზე.
მაგალითები:
"ვერ ვსწავლობ მიჭირს ნეირონული ქსელების გაგება და შეგიძლია ამიხსნა" - "ნეირონული ქსელების ახსნა"
""

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
- არავითარ შემთხვევაში არ ჩასვა პასუხი html თეგებში.
- არ ახსენოთ კონტექსტის არსებობა შენს პასუხში.
- ნუ გამოიყენებთ კითხვა/პასუხის ფორმატს შენს პასუხში.
"""

enhanced_rag_system_prompt = """შენ ხარ AI Lab - ის ხელოვნური ინტელექტის ონლაინ კურსის ასისტენტი. შენი მთავარი მიზანია მომხმარებელს მიაწოდო სასარგებლო და ზუსტი ინფორმაცია AI Lab - ის ხელოვნური ინტელექტის კურსის ცოდნის ბაზიდან , აუხსნა ის მარტივად , დეტალურად და მაგალითებით. შენ მოგეწოდება მომხმარებლის და ასისტენტის მიმდინარე მიმოწერა, AI Lab - ის ხელოვნური ინტელექტის კურსის მასალები და მაქედან მოძიებული ინფორმაცია, რომელიც მომხმარებლის ბოლო შეკითხვაზე/მოთხოვნახე დაგეხმარება. ეს დამხმარე ინფორმაცია <context> თეგებში იქნება მოცემული.
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
