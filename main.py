import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from tavily import TavilyClient

app = FastAPI(
    title="Navier-Stokes Millennium Research Agent",
    description="API Agenta AI wyszukującego i analizującego stan badań nad Problemem Milenijnym Naviera-Stokesa."
)

# Inicjalizacja klientów przy starcie serwera
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Brak zmiennej środowiskowej OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

def get_tavily_client():
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Brak zmiennej środowiskowej TAVILY_API_KEY")
    return TavilyClient(api_key=api_key)

# Funkcja przeszukująca sieć (narzędzie dla agenta)
def search_academic_web(query: str) -> str:
    tavily = get_tavily_client()
    # Szukamy również w serwisach naukowych, np. arXiv, ResearchGate, Clay Mathematics Institute
    response = tavily.search(
        query=f"Navier-Stokes millennium problem {query}",
        search_depth="advanced",
        max_results=5
    )
    
    results = []
    for item in response.get("results", []):
        results.append(f"Tytuł: {item['title']}\nURL: {item['url']}\nTreść: {item['content']}\n")
    
    return "\n---\n".join(results)

# Definicja narzędzia w formacie OpenAI JSON Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_academic_web",
            "description": "Przeszukuje publikacje naukowe i sieć pod kątem problemu Naviera-Stokesa, gładkości i istnienia rozwiązań w 3D.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Słowa kluczowe dotyczące problemu, np. 'blow-up solutions 3D Euler', 'Terence Tao Navier-Stokes', 'Clay Mathematics Institute status'.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# Pętle badawcze agenta
def execute_research(aspect_query: str) -> str:
    client = get_openai_client()
    
    messages = [
        {
            "role": "system",
            "content": (
                "Jesteś wyspecjalizowanym agentem badawczym AI w dziedzinie fizyki matematycznej i równań różniczkowych cząstkowych (PDE). "
                "Twój cel to analiza obecnego stanu wiedzy na temat Problemu Milenijnego Równań Naviera-Stokesa (Existence and Smoothness). "
                "Skupiaj się na: potencjalnych dowodach, rozwiązaniach słabych (Leray-Hopf), problemie Osobliwości (Blow-up), pracach znanych matematyków (np. Terence Tao) i oficjalnych komunkatach Clay Mathematics Institute. "
                "Zawsze opieraj wnioski na źródłach pobranych z narzędzia `search_academic_web` i podawaj adresy URL."
            )
        },
        {"role": "user", "content": f"Przeprowadź badanie na temat: {aspect_query}"}
    ]

    for _ in range(5):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "search_academic_web":
                    args = json.loads(tool_call.function.arguments)
                    search_res = search_academic_web(args["query"])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": search_res
                    })
        else:
            return msg.content

    return "Przekroczono maksymalną liczbę kroków badawczych."

# Schematy zapytań i odpowiedzi API
class ResearchRequest(BaseModel):
    query: str = Field(
        default="Jaki jest aktualny stan prac nad dowodem na istnienie i gładkość rozwiązań 3D Naviera-Stokesa?",
        description="Szczegółowe pytanie badawcze dotyczące równań Naviera-Stokesa."
    )

class ResearchResponse(BaseModel):
    status: str
    query: str
    report: str

@app.get("/")
def health_check():
    return {"status": "ok", "agent": "Navier-Stokes Research Agent"}

@app.post("/research", response_model=ResearchResponse)
def trigger_research(payload: ResearchRequest):
    try:
        report = execute_research(payload.query)
        return ResearchResponse(status="success", query=payload.query, report=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))