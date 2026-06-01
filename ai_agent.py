# ai_agent.py
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ⚠️ Core config: fill in your OpenAI API Key here
# If you're using an academic mirror, forwarding key, or alternative LLM API, configure base_url accordingly
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"

def generate_market_report(model_name, year, mileage, user_price, predicted_price):
    # 1. Initialize the LLM (using GPT-4o with temperature=0.7 for professional yet dynamic output)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # 2. Define the system persona (System Prompt)
    # This controls the model's tone, professional background, and output format
    system_prompt = (
        "You are a senior German automotive market expert and a master negotiator specializing in high-performance cars "
        "(such as BMW M series, Audi R8, and premium executive wagons).\n"
        "The user will provide you with the vehicle specs, the seller's asking price, and the baseline price predicted "
        "by our machine learning system.\n"
        "Your task is to generate a sharp, professional, and data-driven purchase evaluation report in English.\n"
        "You MUST structure your response into the following 3 distinct sections (use Markdown formatting):\n\n"
        "### 1. Price Valuation & Deal Analysis\n"
        "Analyze whether the seller's price is Overpriced or Underpriced compared to our ML baseline. State the exact premium/discount.\n\n"
        "### 2. Technical Vulnerabilities & Maintenance Risks\n"
        "Based on your professional knowledge, point out common mechanical issues for this specific model at this mileage/age "
        "(e.g., rod bearings for older M5, magnetic ride dampers for R8, oil leaks or air suspension for 3 Series Touring).\n\n"
        "### 3. Tactical Negotiation Strategy\n"
        "Provide 2 concrete, aggressive negotiation angles the buyer can use to lower the price based on the price gap and technical risks."
    )
    
    # 3. Receive real-time variables passed in from the frontend / ML engine (User Prompt)
    user_prompt = (
        "Here are the details of the car I am looking at:\n"
        "- Vehicle Model: {model_name}\n"
        "- First Registration Year: {year}\n"
        "- Current Mileage: {mileage} km\n"
        "- Seller's Asking Price: €{user_price}\n"
        "- ML System Predicted Base Value: €{predicted_price}\n"
    )
    
    # 4. Use LangChain's ChatPromptTemplate to assemble the persona and user input into a standard chat template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])
    
    # 5. Use LangChain's LCEL (LangChain Expression Language) pipe syntax
    # The | operator feeds the assembled prompt directly into the LLM
    chain = prompt | llm
    
    # 6. Invoke the chain and wait for the model to generate the in-depth report
    response = chain.invoke({
        "model_name": model_name,
        "year": year,
        "mileage": mileage,
        "user_price": user_price,
        "predicted_price": predicted_price
    })
    
    # Return the text content produced by the model
    return response.content

# 7. Quick local integration test
if __name__ == "__main__":
    print("Connecting to OpenAI LLM for test analysis...")
    # Assume the ML model predicts a fair value of €82,000, but the seller is asking €89,000 (€7,000 over)
    sample_report = generate_market_report(
        model_name="BMW M5", 
        year=2021, 
        mileage=60000, 
        user_price=89000, 
        predicted_price=82000
    )
    print("\n--- LLM-Generated Professional Report ---\n")
    print(sample_report)
