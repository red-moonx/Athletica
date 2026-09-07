import streamlit as st
import json
import os
import csv
from datetime import datetime
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables (fallback for local testing)
load_dotenv("../.env")

# Try to get API key from Streamlit secrets, fallback to environment variable
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API Key is missing. Please add it to Streamlit Secrets or .env file.")
    st.stop()

# Configuration
CHUNKS_PATH = "data/processed/optimizations/optimization_3/chunks.json"
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
LLM_MODEL = "gpt-4o-mini"

# Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Athletica Assistant",
    page_icon="🏃‍♀️",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Global Typography Override for Streamlit containers */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Target specifically the markdown containers used by Streamlit for text */
    [data-testid="stMarkdownContainer"] p, 
    [data-testid="stMarkdownContainer"] li, 
    [data-testid="stMarkdownContainer"] span {
        font-size: 22px !important;
        line-height: 1.6 !important;
    }

    /* Premium Main Title */
    h1 {
        font-size: 3.5rem !important;
        background: -webkit-linear-gradient(45deg, #FF3366, #FF9933);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        margin-bottom: 0 !important;
    }

    /* Target the actual chat message bubbles */
    [data-testid="stChatMessageContent"] {
        font-size: 22px !important;
    }

    /* Expanders (Bibliography Cards) */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out !important;
        margin-bottom: 10px !important;
    }
    
    [data-testid="stExpander"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3) !important;
        border: 1px solid rgba(255, 153, 51, 0.5) !important;
    }
    
    /* Right column Title */
    h3 {
        font-size: 2.2rem !important;
        color: #FF9933 !important;
        font-weight: 600 !important;
    }
    
    /* Ensure the small caption text isn't too tiny now */
    [data-testid="stCaptionContainer"] p {
        font-size: 16px !important;
        color: #aaaaaa !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization & Caching ---
@st.cache_resource(show_spinner="Loading Embeddings Model...")
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner="Loading and Encoding Knowledge Base (Cloud Mode)...")
def load_and_encode_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    chunk_dict = {c["chunk_id"]: c for c in chunks}
    chunk_ids = list(chunk_dict.keys())
    
    # We need the model to encode
    model_instance = load_embedding_model()
    
    # Extract contents to encode
    contents = [chunk_dict[cid]["content"] for cid in chunk_ids]
    
    # Encode all chunks to a NumPy array
    embeddings = model_instance.encode(contents)
    
    # Normalize embeddings for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    
    return chunk_dict, chunk_ids, normalized_embeddings

# Initialize globals
try:
    model = load_embedding_model()
    chunk_dict, chunk_ids, normalized_embeddings = load_and_encode_chunks()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Error initializing services: {e}")
    st.stop()

# --- Functions ---
def save_feedback(question: str, answer: str, score: int):
    """Appends user feedback to a local CSV file."""
    os.makedirs("data", exist_ok=True)
    file_path = "data/feedback_log.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "answer", "score"])
        writer.writerow([datetime.now().isoformat(), question, answer, score])

def search_knn(query: str, top_k: int = 5):
    """Retrieve chunks using in-memory cosine similarity (NumPy dot product)."""
    # Encode the query
    query_vector = model.encode([query])[0]
    
    # Normalize query vector
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = query_vector / query_norm
        
    # Calculate cosine similarity (dot product of normalized vectors)
    similarities = np.dot(normalized_embeddings, query_vector)
    
    # Get top_k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [chunk_ids[i] for i in top_indices]

def issn_dosage_safety_checker(supplement: str, weight_kg: float, dose: float, unit: str = "mg"):
    """
    Verifies supplement dosage safety against ISSN female athlete recommendations.
    """
    supp = supplement.lower()
    
    if "caffeine" in supp:
        # ISSN recommendation: 3-6 mg/kg
        min_dose = weight_kg * 3
        max_dose = weight_kg * 6
        if dose < min_dose:
            return f"Dose of {dose}{unit} is likely too low to be ergogenic. ISSN recommends {min_dose:.1f}-{max_dose:.1f}mg ({weight_kg}kg x 3-6mg/kg)."
        elif dose > max_dose:
            return f"WARNING: Dose of {dose}{unit} exceeds ISSN recommended upper limit of {max_dose:.1f}mg for {weight_kg}kg bodyweight. Risk of anxiety, jitters, and GI distress."
        else:
            return f"SAFE: Dose of {dose}{unit} falls within the optimal ISSN ergogenic range of {min_dose:.1f}-{max_dose:.1f}mg for {weight_kg}kg bodyweight."
            
    elif "creatine" in supp:
        # ISSN recommendation: 3-5 g/day for maintenance, or 20g/day (4x5g) for loading
        dose_g = dose / 1000 if unit.lower() == "mg" else dose
            
        if dose_g > 25:
            return f"WARNING: Dose of {dose_g}g is extremely high. ISSN recommends max 20g/day (divided in 4 doses) for loading, or 3-5g/day for maintenance."
        elif dose_g >= 10:
            return f"SAFE FOR LOADING: {dose_g}g is appropriate for the loading phase (usually 20g/day divided into 4 doses for 5-7 days). Ensure adequate hydration."
        else:
            return f"SAFE FOR MAINTENANCE: {dose_g}g is well within the ISSN maintenance recommendation of 3-5g/day."
            
    elif "iron" in supp:
        dose_mg = dose * 1000 if unit.lower() == "g" else dose
        if dose_mg > 100:
            return f"WARNING: High iron dose ({dose_mg}mg). High doses can cause severe GI distress and inhibit absorption. Only take under medical supervision."
        else:
            return f"Dose of {dose_mg}mg. Iron supplementation should ideally be guided by blood tests (ferritin). Ensure you take it away from calcium and with Vitamin C."
            
    else:
        return f"Supplement '{supplement}' is not currently in the ISSN safety checker database. Please use the search_medical_literature tool to find information manually."

def calculate_energy_availability(energy_intake_kcal: float, exercise_expenditure_kcal: float, weight_kg: float, body_fat_percentage: float):
    """
    Calculates Energy Availability (EA) to assess RED-S risk.
    EA = (Energy Intake - Exercise Energy Expenditure) / Fat-Free Mass
    """
    ffm = weight_kg * (1 - (body_fat_percentage / 100))
    
    if ffm <= 0:
        return "Error: Invalid weight or body fat percentage."
        
    ea = (energy_intake_kcal - exercise_expenditure_kcal) / ffm
    
    if ea >= 45:
        status = "OPTIMAL: EA is ≥ 45 kcal/kg FFM. This supports optimal health, physiological function, and performance."
    elif ea >= 30:
        status = "SUBOPTIMAL: EA is 30-45 kcal/kg FFM. Safe for short-term weight loss, but may impair recovery if sustained."
    else:
        status = "RED-S RISK: EA is < 30 kcal/kg FFM. This is considered Low Energy Availability (LEA) and puts the athlete at high risk for Relative Energy Deficiency in Sport (RED-S), impacting bone health, menstrual function, and performance."
        
    return f"Calculated FFM: {ffm:.1f} kg. Calculated EA: {ea:.1f} kcal/kg FFM.\nAssessment: {status}"

def agent_loop(question):
    """Run the Agentic RAG loop to answer the user question."""
    system_prompt = """
You are Athletica, an expert sports physiology and nutrition assistant for female athletes.
You have access to a curated database of scientific literature via the `search_medical_literature` tool.
Always use this tool to find evidence before answering.
If the retrieved evidence does not contain the answer, use the tool again with DIFFERENT search terms (e.g. synonyms, broader or narrower concepts).
You can search up to 3 times in a single turn. 
If after searching you still cannot find the answer, explicitly say "I don't know based on the provided evidence."
Always base your final answer strictly on the retrieved context. Be supportive, concise, and professional.
"""
    
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": question}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_medical_literature",
                "description": "Searches the curated medical literature database for female athlete physiology. Use this to retrieve evidence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query, e.g. 'caffeine effects on menstrual cycle' or 'iron supplements RED-S'"
                        },
                    },
                    "required": ["query"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "issn_dosage_safety_checker",
                "description": "Calculates and verifies if a specific supplement dosage is safe and optimal for a female athlete based on ISSN guidelines.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "supplement": {
                            "type": "string",
                            "description": "The name of the supplement (e.g. 'caffeine', 'creatine', 'iron')"
                        },
                        "weight_kg": {
                            "type": "number",
                            "description": "The bodyweight of the athlete in kilograms"
                        },
                        "dose": {
                            "type": "number",
                            "description": "The numerical dose amount being considered"
                        },
                        "unit": {
                            "type": "string",
                            "description": "The unit of the dose (e.g. 'mg', 'g')"
                        }
                    },
                    "required": ["supplement", "weight_kg", "dose", "unit"],
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_energy_availability",
                "description": "Calculates Energy Availability (EA) to assess RED-S risk based on calories and body composition.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "energy_intake_kcal": {
                            "type": "number",
                            "description": "Total daily calorie intake in kcal"
                        },
                        "exercise_expenditure_kcal": {
                            "type": "number",
                            "description": "Calories burned strictly through exercise in kcal"
                        },
                        "weight_kg": {
                            "type": "number",
                            "description": "The bodyweight of the athlete in kilograms"
                        },
                        "body_fat_percentage": {
                            "type": "number",
                            "description": "The body fat percentage of the athlete (e.g. 20 for 20%)"
                        }
                    },
                    "required": ["energy_intake_kcal", "exercise_expenditure_kcal", "weight_kg", "body_fat_percentage"],
                }
            }
        }
    ]
    
    max_iterations = 4
    all_citations = []
    used_tools = []
    
    status_container = st.empty()
    
    for i in range(max_iterations):
        with status_container.container():
            st.info(f"🤔 Thinking... (Iteration {i+1}/{max_iterations})")
            
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
            temperature=0.0
        )
        msg = response.choices[0].message
        
        # Manually convert message object to dict for robustness
        msg_dict = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": t.id,
                    "type": t.type,
                    "function": {"name": t.function.name, "arguments": t.function.arguments}
                } for t in msg.tool_calls
            ]
        messages.append(msg_dict)
        
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                used_tools.append(tool_call.function.name)
                
                if tool_call.function.name == "search_medical_literature":
                    args = json.loads(tool_call.function.arguments)
                    search_query = args.get("query", "")
                    
                    with status_container.container():
                        st.info(f"🔍 Searching literature for: **'{search_query}'**...")
                        
                    retrieved_ids = search_knn(search_query, top_k=5)
                    new_citations = [chunk_dict[cid] for cid in retrieved_ids]
                    
                    # Prevent duplicates in citations while preserving order
                    existing_pmids_chunks = set(c["chunk_id"] for c in all_citations)
                    for c in new_citations:
                        if c["chunk_id"] not in existing_pmids_chunks:
                            all_citations.append(c)
                            existing_pmids_chunks.add(c["chunk_id"])
                    
                    context_text = "\n\n".join([f"[{c['pmid']}] {c['content']}" for c in new_citations])
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": context_text
                    })
                elif tool_call.function.name == "issn_dosage_safety_checker":
                    args = json.loads(tool_call.function.arguments)
                    supplement = args.get("supplement", "unknown")
                    weight_kg = args.get("weight_kg", 60.0)
                    dose = args.get("dose", 0.0)
                    unit = args.get("unit", "mg")
                    
                    with status_container.container():
                        st.info(f"🧮 Calculating ISSN safety for {dose}{unit} of {supplement} at {weight_kg}kg...")
                        
                    result = issn_dosage_safety_checker(supplement, weight_kg, dose, unit)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                elif tool_call.function.name == "calculate_energy_availability":
                    args = json.loads(tool_call.function.arguments)
                    ei = args.get("energy_intake_kcal", 0)
                    eee = args.get("exercise_expenditure_kcal", 0)
                    weight = args.get("weight_kg", 0)
                    bf = args.get("body_fat_percentage", 0)
                    
                    with status_container.container():
                        st.info(f"⚖️ Calculating Energy Availability (EA) for {weight}kg ({bf}% fat)...")
                        
                    result = calculate_energy_availability(ei, eee, weight, bf)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
        else:
            # Final answer
            status_container.empty()
            return msg.content, all_citations, i + 1, used_tools
            
    status_container.empty()
    return "I couldn't find a definitive answer after multiple searches.", all_citations, max_iterations, used_tools

def render_bibliography(citations):
    """Render the bibliography dynamically grouped by paper title."""
    if not citations:
        st.info("No medical evidence retrieved for this query.")
        return
        
    st.markdown("### 📚 Medical Evidence")
    st.caption("Source documents used to generate the last response.")
    
    # Group by title
    grouped = defaultdict(list)
    for c in citations:
        grouped[c["title"]].append(c)
        
    for title, chunks in grouped.items():
        with st.expander(f"📖 {title}"):
            authors = chunks[0]["authors"]
            pmid = chunks[0]["pmid"]
            st.markdown(f"**Authors:** {authors}")
            st.markdown(f"**PMID:** [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
            st.divider()
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"**Excerpt {i}:**")
                st.write(chunk['content'])
                st.divider()

# --- UI Setup ---
with st.sidebar:
    st.markdown("### ⚙️ Admin & Logs")
    if os.path.exists("data/feedback_log.csv"):
        with open("data/feedback_log.csv", "r", encoding="utf-8") as f:
            csv_data = f.read()
        st.download_button(
            label="📥 Descargar Feedback CSV",
            data=csv_data,
            file_name="feedback_log.csv",
            mime="text/csv"
        )
    else:
        st.info("Aún no hay feedback registrado.")

# Two-column layout
col1, col2 = st.columns([6, 4], gap="large")

with col1:
    st.title("🏃‍♀️ Athletica")
    st.markdown("**Evidence-based Nutrition & Physiology Assistant for Female Athletes**")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am Athletica. Ask me any question about female athlete physiology, nutrition, the menstrual cycle, or RED-S."}
        ]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("E.g., How does the menstrual cycle affect my training?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # 3. Generate Answer using Agentic RAG Loop
            answer, citations, iterations, used_tools = agent_loop(prompt)
            
            # Display answer
            st.markdown(answer)

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "citations": citations,
            "iterations": iterations,
            "used_tools": used_tools
        })
        
        # Rerun to update the right column immediately
        st.rerun()

with col2:
    # Always display the citations from the LAST assistant message that has them
    last_citations = None
    last_iterations = None
    last_tools = []
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and "citations" in msg:
            last_citations = msg["citations"]
            last_iterations = msg.get("iterations", 1)
            last_tools = msg.get("used_tools", [])
            break
            
    if last_citations is not None:
        render_bibliography(last_citations)
        
        st.divider()
        st.markdown("### 🤖 Agent Stats")
        
        # Display iterations with a nice visual badge-like info box
        if last_iterations == 1:
            st.success(f"**Reasoning Iterations:** {last_iterations} (One-shot success)")
        else:
            st.warning(f"**Reasoning Iterations:** {last_iterations} (Iterative search used)")
            
        # Display Tools used
        if last_tools:
            st.markdown("**Tools Invoked:**")
            unique_tools = set(last_tools)
            for t in unique_tools:
                count = last_tools.count(t)
                icon = "🔧"
                desc = ""
                if t == "search_medical_literature": 
                    icon = "🔍"
                    desc = "Queried the in-memory medical vector database"
                elif t == "calculate_energy_availability": 
                    icon = "⚖️"
                    desc = "Calculated RED-S risk based on calories and FFM"
                elif t == "issn_dosage_safety_checker": 
                    icon = "🧮"
                    desc = "Verified supplement dosage against ISSN guidelines"
                
                st.markdown(f"- {icon} `{t}` (x{count})")
                if desc:
                    st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;*{desc}*")
        else:
            st.markdown("**Tools Invoked:** None")
            
        # Feedback UI
        st.divider()
        st.markdown("### 📝 Rate this answer")
        
        if len(st.session_state.messages) >= 2:
            last_user_msg = st.session_state.messages[-2]["content"]
            last_asst_msg = st.session_state.messages[-1]["content"]
            
            fb_col1, fb_col2 = st.columns(2)
            msg_index = len(st.session_state.messages)
            
            with fb_col1:
                if st.button("👍 Helpful", key=f"up_{msg_index}"):
                    save_feedback(last_user_msg, last_asst_msg, 1)
                    st.success("Feedback saved!")
            with fb_col2:
                if st.button("👎 Not Helpful", key=f"down_{msg_index}"):
                    save_feedback(last_user_msg, last_asst_msg, -1)
                    st.success("Feedback saved!")
                    
    else:
        st.markdown("### 📚 Medical Evidence")
        st.info("Ask a question in the chat to view the retrieved scientific articles.")
