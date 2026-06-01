# ai_agent.py
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ⚠️ 核心配置：在这里填入你的 OpenAI API Key 
# 如果你使用的是学术镜像、转发 Key 或者其他大模型 API，可以配置 base_url
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"

def generate_market_report(model_name, year, mileage, user_price, predicted_price):
    # 1. 初始化大模型 (使用 GPT-4o，将 temperature 设为 0.7 保持专业又不失灵动)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # 2. 设定“懂车帝”的系统人设（System Prompt）
    # 这决定了大模型的说话语气、专业背景和输出格式
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
    
    # 3. 接收从网页端/机器学习引擎传过来的实时变量（User Prompt）
    user_prompt = (
        "Here are the details of the car I am looking at:\n"
        "- Vehicle Model: {model_name}\n"
        "- First Registration Year: {year}\n"
        "- Current Mileage: {mileage} km\n"
        "- Seller's Asking Price: €{user_price}\n"
        "- ML System Predicted Base Value: €{predicted_price}\n"
    )
    
    # 4. 使用 LangChain 的 ChatPromptTemplate 将人设和用户输入拼装成一个标准的对话模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])
    
    # 5. 使用 LangChain 标志性的 LCEL（LangChain Expression Language）链式语法
    # 管道符 | 意思是把拼装好的 prompt 顺着管道直接喂给大模型
    chain = prompt | llm
    
    # 6. 一键启动，等待大模型生成深度报告
    response = chain.invoke({
        "model_name": model_name,
        "year": year,
        "mileage": mileage,
        "user_price": user_price,
        "predicted_price": predicted_price
    })
    
    # 返回大模型吐出来的文本内容
    return response.content

# 7. 本地快速联调测试
if __name__ == "__main__":
    print("正在连接 OpenAI 大模型进行测试分析...")
    # 假设精算师算出来值 82000，但卖家要 89000（贵了 7000 欧）
    sample_report = generate_market_report(
        model_name="BMW M5", 
        year=2021, 
        mileage=60000, 
        user_price=89000, 
        predicted_price=82000
    )
    print("\n--- 大模型生成的专业报告 ---\n")
    print(sample_report)