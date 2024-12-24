import operator
import re
from typing import Optional, Callable, Any, Awaitable
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
from pinecone import Pinecone
from open_webui.utils.task import replace_prompt_variable, replace_messages_variable

EmitterType = Optional[Callable[[dict], Awaitable[None]]]

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
        print(f"{self.valves.PIPELINE_ID}, __init__,  {self.valves=}")
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

        agent = AILabAgent(self.valves, __user__)

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

        self.pinecone_client = Pinecone(api_key="pcsk_5o8tFS_MiBEvmq8wNjo4GYvMB7yydCV1Gwaj4PT3b6VF7QzZaWKSR2QYb2F4axChcrkSj8")
        self.index = self.pinecone_client.Index("ai-lab-academy-courses")

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
            streaming=True,
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
            """
            Search the AiLab knowledge base using a semantic search query.

            Args:
                query (str): The search query to find relevant information in AiLab's knowledge base

            Returns:
                list: A list of relevant text snippets from the knowledge base. Returns an error message
                     if the search fails or no results are found.
            """
            try:
                print(f"{self.valves.PIPELINE_ID}, ailab_knowledge_search,  Searching with query: {query}")
                embedding_client = OpenAI(
                    base_url="https://api.deepinfra.com/v1/openai",
                    api_key="sniRj9wr02CNUPLnq548PQAwnKNnFfK3",
                )
                query_embedding = (
                    embedding_client.embeddings.create(
                        model="BAAI/bge-m3",
                        input=query,
                        encoding_format="float"
                    )
                    .data[0]
                    .embedding
                )

                results = self.index.query(
                    namespace="notebooks",
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True
                )

                if not results['matches']:
                    return ["ინფორმაცია ვერ მოიძებნა."]

                formatted_results = []
                for match in results['matches']:
                    if match['metadata'] and "content" in match['metadata']:
                        formatted_results.append(match['metadata']["content"])

                return (
                    formatted_results
                    if formatted_results
                    else ["ინფორმაცია ვერ მოიძებნა."]
                )

            except Exception as e:
                print(f"{self.valves.PIPELINE_ID}, ailab_knowledge_search, Error: {str(e)}")
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
        print(f"{self.valves.PIPELINE_ID}, run_ailab_knowledge_search, START")
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
            print(f"{self.valves.PIPELINE_ID}, run_ailab_knowledge_search, Searching for query: {query}")
            embedding_client = OpenAI(
                base_url="https://api.deepinfra.com/v1/openai",
                api_key="sniRj9wr02CNUPLnq548PQAwnKNnFfK3",
            )

            query_embedding = (
                embedding_client.embeddings.create(
                    model="BAAI/bge-m3",
                    input=query,
                    encoding_format="float"
                )
                .data[0]
                .embedding
            )

            results = self.index.query(
                namespace="notebooks",
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )

            formatted_results = []
            used_files = set()

            for match in results['matches']:
                if not match['metadata'] or "text" not in match['metadata']:
                    continue

                file_name = match['metadata'].get("file", "")
                if file_name in used_files:
                    continue

                used_files.add(file_name)
                content = match['metadata']["text"]
                content = content.replace("Markdown:", "").replace("<br/>", "\n")
                content = re.sub(r"#*\s*", "", content)
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
        print(f"{self.valves.PIPELINE_ID}, run_query_rewriter, rewritten_querry={messages.content}")
        return {"rewritten_query": messages.content}

    def run_final_answer(self, state):
        print(f"{self.valves.PIPELINE_ID}, run_final_answer, START")
        query = state["query"]
        context = state["intermediate_steps"][-1].content
        if context:
            print("> if context")
            chat_history = build_rag_chat_history(context, query, state["messages"])
        else:
            chat_history = state["messages"]
        print(f"{self.valves.PIPELINE_ID}, run_query_rewriter,{chat_history=}")
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
        print("build_rag_chat_history , start")
        system_message = SystemMessage(content=enhanced_rag_system_prompt)
        chat_history = [
            message for message in messages if not isinstance(message, SystemMessage)
        ]
        chat_history = convert_langchain_to_openai(chat_history)
        user_message_content = replace_messages_variable(rag_user_prompt, chat_history)
        user_message_content = user_message_content.format(context=context, query=query)
        user_message = HumanMessage(content=user_message_content)
        print("build_rag_chat_history, end")
        return [system_message, user_message]
    except Exception as e:
        print(f"Error in build_rag_chat_history: {str(e)}")
        return messages


rewrite_template = """ 
კარგად განაალიზე მომხმარებელის შეკთხვა ეხება თუ არა AI Lab - ის ხელოვნური ინტელექტის კურს. 
გაითვალისწინე:
1. თუ მომხმარებლის შეკითხვა არ ეხება AI Lab-ის ხელოვნური ინტელექტის კურსს, დააბრუნე მხოლოდ გადაწერილი query - "არა_რელევანტური".
2. თუ შეკითხვა ეხება AI Lab-ის ხელოვნური ინტელექტის კურსს,მომხმარებლის შეკითხვისა და მიმოწერის გათვალისწინებით, გადააკეთე მომხმარებლის შეკითხვა AI Lab-ის ხელოვნური ინტელექტის კურსის ინფორმაციის მოსაძებნად. შენ მიერ გადაწერილი საძიებო ტექსტი უნდა იყოს ამომწურავი მაგრამ ლაკონური, მხოლოდ მომხმარებლის ბოლო მესიჯში ნაგულისხმევ ინფორმაციით.

დააბრუნე მხოლოდ გადაწერილი query.

მაგალითები:
"ვერ ვსწავლობ მიჭირს ნეირონული ქსელების გაგება და შეგიძლია ამიხსნა" - "ნეირონული ქსელების ახსნა"
"როგორ მუშაობს ნეირონული ქსელი?" - "ნეირონული ქსელის მუშაობის პრინციპი"
"სწავლებაში გამომადგება ეს კურსი?" - "AI Lab-ის კურსის სასწავლო შედეგები"
"რა არის პერცეპტრონი" - "პერცეპტრონის განმარტება და გამოყენება"
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
- უპასუხე მხოლოდ კონტექსტში მოცემული ინფორმაციის მიხედვით
- არ ჩასვა პასუხი html თეგებში
- არ ახსენო კონტექსტის არსებობა შენს პასუხში
- ნუ გამოიყენებ კითხვა/პასუხის ფორმატს
- მომხმარებელს მიაწოდე დეტალური და მკაფიო პასუხი მარტივი ენით
- გამოიყენე მაგალითები პასუხის უკეთ ასახსნელად

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

enhanced_rag_system_prompt = """შენ ხარ AI Lab-ის ხელოვნური ინტელექტის ასისტენტი. შენი მიზანია მომხმარებლებს მიაწოდო სასარგებლო და ზუსტი ინფორმაცია AI Lab-ის კურსის მასალებიდან, აუხსნა კონცეფციები დეტალურად და მარტივი მაგალითებით. 
გაითვალისწინე , რომ თუ მომხმარებლის შეკითხვა არ არის რელევანტური AI Lab-ის ხელოვნური ინტელექტის კურსთან მიმართებაში:
უპასუხე:"ბოდიშს გიხდით, მე ვარ AI Lab-ის ხელოვნური ინტელექტის კურსის ასისტენტი და შემიძლია დაგეხმაროთ მხოლოდ კურსთან დაკავშირებულ საკითხებში. თქვენს შეკითხვაზე პასუხის გასაცემად გთხოვთ მიმართოთ შესაბამის სერვისს."

შენ მოგეწოდება:
1. მომხმარებლის და ასისტენტის მიმოწერა
2. AI Lab-ის ხელოვნური ინტელექტის კურსის მასალები
3. მოძიებული რელევანტური ინფორმაცია კონტექსტში

დაიცავი შემდეგი წესები:
- უპასუხე მხოლოდ კონტექსტში მოცემული ინფორმაციის მიხედვით
- არ ახსენო კონტექსტის არსებობა შენს პასუხში
- ნუ გამოიყენებ კითხვა/პასუხის ფორმატს
- გამოიყენე მხოლოდ რელევანტური ინფორმაცია მასალებიდან
- არ უპასუხო შეკითხვებს, რომლებიც არ ეხება AI Lab-ის კურსს
- მიაწოდე ზუსტი, დეტალური და ამომწურავი პასუხები
- არ გამოიყენო bullet point-ები და **bold** ფორმატირება
- თუ შეკითხვა არ არის კონკრეტული, სთხოვე დააკონკრეტოს რა აინტერესებს
- გამოიყენე მარტივი ენა და მაგალითები ახსნისას

პასუხის გაცემისას:
1. დარწმუნდი რომ პასუხი ეხება AI Lab-ის კურსს
2. გამოიყენე მხოლოდ კონტექსტში მოცემული ინფორმაცია
3. აუხსენი კონცეფციები მარტივი და გასაგები ენით
4. მოიყვანე პრაქტიკული მაგალითები უკეთ გასაგებად
5. დაეხმარე მომხმარებელს კონცეფციების სიღრმისეულ გაგებაში"""

rag_user_prompt = """მიმდინარე მიმოწერა მომხმარებელსა და AI Lab - ის ციფრულ ასისტენტს შორის:
{{MESSAGES:END:5}}

<context>
{context}
</context>

უპასუხე მომხმარებლის ბოლო მესიჯს:
{query}"""