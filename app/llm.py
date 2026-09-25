from langchain_groq import ChatGroq
import yaml
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate

from app.schemas import ClassificationResult, PromptConfig
from pydantic import BaseModel
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

# Pass the provider/model slug exactly as listed on OpenRouter
# llm  = ChatOpenRouter(model="qwen/qwen3.8-27b:free",base_url="https://openrouter.ai/api/v1")


class ClassificationResult(BaseModel):
    category: str
    summary: str

def load_prompt(
    prompt_file: str = "prompts/v1.yaml",) -> PromptConfig:
    with open(prompt_file, "r") as f:
        prompt_data = yaml.safe_load(f)
    return PromptConfig(
        version=prompt_data["version"],
        system_prompt=prompt_data["system_prompt"]
    )

def classify_email(email:str) -> ClassificationResult:
    prompt_config = load_prompt()
    structured_llm = llm.with_structured_output(ClassificationResult, method= "json_mode")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt_config.system_prompt),
        ("user", "{email_content}")
    ])
    chain = prompt_template | structured_llm
    response = chain.invoke({"email_content": email})
    return response

if __name__ == "__main__":
    email = "I was charged twice for my  spotify's subscription this month."

    result = classify_email(email)

    print(result.model_dump())