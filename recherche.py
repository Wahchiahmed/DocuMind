import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

print("🔄 Initialisation du système complet DocuMind...")

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if not os.path.exists("base_documind"):
    raise FileNotFoundError("⚠ La base vectorielle 'base_documind' est introuvable. Lancez main.py d'abord.")
if "GROQ_API_KEY" not in os.environ:
    raise EnvironmentError("⚠️ Clé API Groq manquante. Ajoutez-la dans votre fichier .env")

vectorstore = FAISS.load_local("base_documind", embeddings_model, allow_dangerous_deserialization=True)


llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

print("✅ Base de données et LLM prêts !")
print("-" * 50)


question = "How to navigate directories and list files in Linux?"
print(f"👤 Utilisateur : '{question}'")

resultats = vectorstore.similarity_search(question, k=3)
contexte = "\n\n".join([doc.page_content for doc in resultats])

prompt_template = """
Tu es un assistant technique expert en administration système Linux.
Utilise UNIQUEMENT le contexte fourni ci-dessous pour répondre à la question de l'utilisateur.
Si tu ne trouves pas la réponse dans le contexte, dis simplement "Je n'ai pas cette information dans ma documentation." Ne cherche pas à deviner.
Fais une réponse structurée, utilise des puces si besoin, et sois clair. Tu peux répondre en français.

Contexte extrait des manuels :
{contexte}

Question : {question}

Réponse :
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["contexte", "question"])
chain = prompt | llm

print("\n🤖 L'IA lit la documentation et rédige la réponse...")
reponse = chain.invoke({"contexte": contexte, "question": question})

print("\n" + "=" * 50)
print("🎯 RÉPONSE DE DOCUMIND :")
print("=" * 50)
print(reponse.content)