## 1. AI Agent 기초

### 1.1 Tool(도구)의 핵심 개념

**Tool의 3가지 구성요소:**
1. **name**: 도구를 식별하는 고유한 이름
2. **description**: 도구의 기능과 사용 시점을 설명 (가장 중요!)
3. **args_schema**: 입력 파라미터의 형식 정의

### 1.2 사용자 정의 도구 생성

```python
from langchain.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """두 숫자를 더합니다. 수학적 덧셈이 필요할 때 사용하세요."""
    return a + b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """두 숫자를 곱합니다. 수학적 곱셈이 필요할 때 사용하세요."""
    return a * b
```

### 1.3 도구 바인딩

**LLM에 도구 연결하기:**
```python
from langchain.chat_models import init_chat_model

# 모델 초기화
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# 도구 바인딩
llm_with_tools = llm.bind_tools([add_numbers, multiply_numbers])
```

### 1.4 Agent 구조

**Agent의 핵심 구성요소:**
1. **LLM**: Agent의 두뇌 역할 (추론 및 의사결정)
2. **Tools**: 실제 작업을 수행하는 함수들
3. **Agent Executor**: LLM과 도구 간의 상호작용 관리

**Agent 생성 과정:**
```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are very powerful assistant"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Agent 생성
agent = create_tool_calling_agent(llm, tools, prompt)

# Agent Executor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
```

### 1.5 주요 Built-in Tools

#### TavilySearchResults
- AI Agent 전용 검색 엔진
- 실시간 정보 검색 가능
- 주요 파라미터: `max_results`, `search_depth`, `include_answer`

#### PythonREPLTool
- 실시간 Python 코드 실행
- 복잡한 계산 및 데이터 처리
- 동적 프로그래밍 로직 수행

### 1.6 ReAct Framework

**ReAct = Reasoning + Acting**

**동작 과정:**
1. **Question**: 사용자 질문 입력
2. **Thought**: 어떤 도구를 사용할지 추론
3. **Action**: 도구 선택 및 실행
4. **Observation**: 도구 실행 결과 관찰
5. **반복**: 최종 답변까지 반복

---

## 2. LangGraph 개념과 활용

### 2.1 LangGraph란?

**정의:** 대규모 언어 모델(LLM)의 워크플로우를 그래프 기반으로 설계하고 실행할 수 있게 해주는 프레임워크

**기존 방식과의 차이점:**
| 구분 | 기존 방식 | LangGraph |
|------|-----------|-----------|
| 구조 | 선형적 | 그래프 기반 |
| 분기 처리 | 제한적 | 유연한 조건부 분기 |
| 상태 관리 | 어려움 | 내장된 상태 관리 |
| 재사용성 | 낮음 | 높음 (모듈화) |
| 디버깅 | 어려움 | 단계별 추적 가능 |

### 2.2 핵심 구성요소

#### 1. State (상태)
```python
from typing import TypedDict

class State(TypedDict):
    input_text: str
    processed_text: str
    word_count: int
```

#### 2. Node (노드)
```python
def text_processor_node(state: State) -> State:
    # 1. 상태에서 데이터 추출
    input_text = state["input_text"]
    
    # 2. 작업 수행
    processed_text = input_text.upper()
    
    # 3. 결과를 상태에 저장
    state["processed_text"] = processed_text
    
    return state
```

#### 2. Edge (엣지)
- 노드 간의 연결과 흐름 정의
- 조건부 분기 지원

### 2.3 기본 그래프 구성

```python
from langgraph.graph import StateGraph, START, END

# 그래프 생성
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("text_processor", text_processor_node)
workflow.add_node("word_counter", word_counter_node)

# 엣지 설정
workflow.add_edge(START, "text_processor")
workflow.add_edge("text_processor", "word_counter")
workflow.add_edge("word_counter", END)

# 그래프 컴파일
graph = workflow.compile()
```

### 2.4 ChatBot 구현

#### add_messages 함수
- 메시지 리스트를 효율적으로 관리
- 동일한 ID의 메시지는 새로운 메시지로 대체

```python
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

#### ChatBot Node
```python
def chatbot_node(state: State):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

### 2.5 ToolNode 활용

**ToolNode의 역할:**
- LLM이 생성한 도구 호출 요청을 자동 감지
- 해당 도구를 실제로 실행
- 결과를 ToolMessage 형태로 상태에 추가

```python
from langgraph.prebuilt import ToolNode, tools_condition

# ToolNode 생성
tool_node = ToolNode(tools=tools)

# 조건부 엣지로 도구 사용 제어
graph_builder.add_conditional_edges(
    "chatbot_node",
    tools_condition  # tool_calls가 있으면 "tools"로, 없으면 "END"로
)
```

### 2.6 Memory 기능

#### MemorySaver
- 대화 기록을 메모리에 저장
- thread_id별로 세션 관리

```python
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Memory 설정
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# 사용자별 세션 관리
user_config = RunnableConfig(
    recursion_limit=10,
    configurable={"thread_id": "user_session_1"}
)

# 대화 실행
result = graph.stream({"messages": [("user", "안녕하세요")]}, config=user_config)
```

