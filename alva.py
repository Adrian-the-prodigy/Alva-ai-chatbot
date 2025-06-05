
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS
from transformers import pipeline
import gradio as gr

# Reasoning Neural Network (input: 768 = 384 + 384)
class ReasoningNet(nn.Module):
    def __init__(self):
        super(ReasoningNet, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

class ContextualLearningBot:
    def __init__(self, name="SmartBot"):
        self.name = name
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.reasoner = ReasoningNet()
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

        self.memory = []
        self.user_id = None
        self.memory_file = None
        self.master_memory_file = "ALVA_ALL.pkl"

    def get_memory_file(self, user_id):
        safe_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_")).rstrip()
        return self.master_memory_file if safe_id == "ALVA_ALL" else f"bot_memory_{safe_id}.pkl"

    def load_memory(self, user_id):
        self.user_id = user_id
        self.memory_file = self.get_memory_file(user_id)
        self.memory = []
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "rb") as f:
                self.memory = pickle.load(f)

    def save_memory(self):
        if self.memory_file:
            with open(self.memory_file, "wb") as f:
                pickle.dump(self.memory, f)

    def add_to_master_memory(self, new_entry):
        master_mem = []
        if os.path.exists(self.master_memory_file):
            with open(self.master_memory_file, "rb") as f:
                master_mem = pickle.load(f)
        if new_entry['question'] not in {e['question'] for e in master_mem}:
            master_mem.append(new_entry)
            with open(self.master_memory_file, "wb") as f:
                pickle.dump(master_mem, f)

    def find_best_match(self, user_input):
        if not self.memory:
            return None, 0.0
        input_embedding = self.model.encode(user_input, convert_to_tensor=True)
        memory_embeddings = [item["embedding"] for item in self.memory]
        similarities = util.pytorch_cos_sim(input_embedding, torch.stack(memory_embeddings))[0]
        best_score, best_idx = torch.max(similarities, dim=0)
        return (self.memory[best_idx], best_score.item()) if best_score >= 0.75 else (None, best_score.item())

    def search_web_summary(self, query):
        snippets = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=19):
                    if r and r.get("body"):
                        snippets.append(r["body"])
        except Exception as e:
            print(f"[Web Search Error] {e}")
        combined = " ".join(snippets)[:9000]
        if not combined:
            return "No web info found."
        try:
            summary = self.summarizer(combined, max_length=1000, min_length=700, do_sample=False)[0]["summary_text"]
            return summary
        except Exception as e:
            print(f"[Summarization Error] {e}")
            return combined

    def get_answer_from_web(self, question, context):
        try:
            return self.qa_pipeline({"question": question, "context": context})["answer"]
        except Exception as e:
            print(f"[QA Error] {e}")
            return "No answer extracted."

    def generate_final_response(self, user_response, web_response, reasoning_score, similarity_score):
        web_response = web_response.strip().capitalize().rstrip(".") + "."
        user_response = user_response.strip().capitalize().rstrip(".") + "."

        if similarity_score > 0.85 and reasoning_score > 0.6:
            return f"{user_response} {web_response} It all fits together pretty well."
        elif similarity_score > 0.5:
            return f"{user_response} I found online sources that mention  {web_response.lower()}"
        else:
            return f"{user_response} That said, online sources describe it as {web_response.lower()}"

    def auto_learn_if_disliked(self, user_input):
        web_summary = self.search_web_summary(user_input)
        best_web_answer = self.get_answer_from_web(user_input, web_summary)

        input_embed = self.model.encode(user_input, convert_to_tensor=True)
        web_embed = self.model.encode(best_web_answer, convert_to_tensor=True)

        new_entry = {
            "question": user_input,
            "response": best_web_answer,
            "embedding": input_embed,
            "web_summary": web_summary,
            "web_best_answer": best_web_answer,
            "match": 1.0,
            "reasoning": 1.0
        }

        self.memory.append(new_entry)
        self.save_memory()
        return best_web_answer

    def learn_response(self, user_input, user_answer):
        web_summary = self.search_web_summary(user_input)
        best_web_answer = self.get_answer_from_web(user_input, web_summary)

        input_embed = self.model.encode(user_input, convert_to_tensor=True)
        user_embed = self.model.encode(user_answer, convert_to_tensor=True)
        web_embed = self.model.encode(best_web_answer, convert_to_tensor=True)

        combined_embed = torch.cat((user_embed, web_embed), dim=0).unsqueeze(0)
        reasoning_score = self.reasoner(combined_embed).item()
        similarity = util.pytorch_cos_sim(user_embed, web_embed).item()

        new_entry = {
            "question": user_input,
            "response": user_answer,
            "embedding": input_embed,
            "web_summary": web_summary,
            "web_best_answer": best_web_answer,
            "match": similarity,
            "reasoning": reasoning_score
        }

        self.memory.append(new_entry)
        self.save_memory()
        if self.user_id != "ALVA_ALL":
            self.add_to_master_memory(new_entry)

        return (
            f"**Web Best Response**: {best_web_answer}\n\n"
            f"**Similarity with your answer**: {similarity * 100:.2f}%\n"
            f"**Reasoning Confidence**: {reasoning_score:.2f}\n\n"
            f"📚 Learned this new info and updated memory.\n\n"
            f"{self.generate_final_response(user_answer, best_web_answer, reasoning_score, similarity)}"
        )

    def get_response(self, user_input):
        match, _ = self.find_best_match(user_input)
        if match:
            return self.generate_final_response(
                match["response"],
                match["web_best_answer"],
                match["reasoning"],
                match["match"]
            )
        return None

# Instantiate bot
bot = ContextualLearningBot()

# Gradio function
def chat_gradio(user_id, user_input, user_teach=None, feedback="like"):
    bot.load_memory(user_id)
    response = bot.get_response(user_input)

    if response:
        if feedback == "dislike" and user_teach:
            learn_msg = bot.learn_response(user_input, user_teach.strip())
            return learn_msg, ""
        return response, ""

    if user_teach and user_teach.strip():
        learn_msg = bot.learn_response(user_input, user_teach.strip())
        return learn_msg, ""

    web_auto = bot.auto_learn_if_disliked(user_input)
    return f"Here's what I found online: {web_auto}\n\n(If you disagree, please teach me)", ""

# Gradio UI
demo = gr.Interface(
    fn=chat_gradio,
    inputs=[
        gr.Textbox(label="User ID", value="general"),
        gr.Textbox(label="Ask me anything"),
        gr.Textbox(label="Teach me (if I don't know) (optional)"),
        gr.Radio(["like", "dislike"], label="Do you like the answer?", value="like")
    ],
    outputs=[
        gr.Textbox(label="Bot Response", lines=8),
        gr.Textbox(visible=False)
    ],
    title="ALVA",
    description="AI chatbot that learns from you and the web."
)

if __name__ == "__main__":
    demo.launch()
