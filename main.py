
import os
import sys
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    print(f"""
{BLUE}{BOLD}
╔══════════════════════════════════════════════╗
║         🐧 DocuMind - Vector Base Builder        ║
║     Linux & Cisco RAG Assistant - v2.0       ║
╚══════════════════════════════════════════════╝
{RESET}""")


def charger_tous_les_pdfs(dossier: str) -> list:

    pdf_files = glob.glob(os.path.join(dossier, "*.pdf"))

    if not pdf_files:
        print(f"{RED}❌ No PDF found in '{dossier}/' folder.{RESET}")
        print(f"{YELLOW}👉 Add your PDFs to the 'data/' folder and rerun.{RESET}")
        sys.exit(1)

    print(f"{BLUE}📂 PDFs found:{RESET}")
    for f in pdf_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"   📄 {os.path.basename(f)} ({size_mb:.1f} MB)")
    print()

    all_documents = []
    for pdf_path in pdf_files:
        try:
            print(f"   ⏳ Loading: {os.path.basename(pdf_path)}...")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            # Add clean source metadata
            for doc in docs:
                doc.metadata["source"] = os.path.basename(pdf_path)
            all_documents.extend(docs)
            print(f"   {GREEN}✅ {len(docs)} pages loaded{RESET}")
        except Exception as e:
            print(f"   {RED}❌ Error loading {pdf_path}: {e}{RESET}")

    return all_documents


def main():
    print_banner()

    if os.path.exists("base_documind"):
        print(f"{YELLOW}⚠️  Vector base 'base_documind' already exists!{RESET}")
        print(f"   Options:")
        print(f"   1) Delete 'base_documind' folder to rebuild from scratch")
        print(f"   2) Use {BOLD}ajouter_document.py{RESET} to add a new PDF")
        print()
        choix = input(f"{BOLD}Rebuild from scratch? (yes/no): {RESET}").strip().lower()
        if choix not in ["yes", "y", "oui", "o"]:
            print(f"{GREEN}✅ Keeping existing base. Exiting.{RESET}")
            sys.exit(0)
        print()

    if not os.path.exists("data"):
        os.makedirs("data")
        print(f"{YELLOW}📁 'data/' folder created. Add your PDFs and rerun.{RESET}")
        sys.exit(1)

    print(f"{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD}📖 STEP 1/4 - Loading PDFs...{RESET}")
    documents = charger_tous_les_pdfs("data")
    print(f"\n{GREEN}✅ Total: {len(documents)} pages loaded{RESET}\n")

    print(f"{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD}✂️  STEP 2/4 - Splitting into chunks...{RESET}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=[
            "\n## ", "\n### ", "\n#### ",
            "\n\n",
            "\n",
            ". ",
            " ", ""
        ]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"   📦 Chunk size    : 1000 characters")
    print(f"   🔁 Chunk overlap : 200 characters")
    print(f"{GREEN}✅ Total: {len(chunks)} chunks created{RESET}\n")

    # ── Step 3: Embeddings Model ───────────────────────────────
    print(f"{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD}🧠 STEP 3/4 - Loading embedding model...{RESET}")
    print(f"   Model: all-MiniLM-L6-v2 (HuggingFace)")

    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"{GREEN}✅ Embedding model ready{RESET}\n")

    print(f"{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD}🏗️  STEP 4/4 - Creating FAISS vector database...{RESET}")
    print(f"{YELLOW}⏳ This may take 2-5 minutes depending on your CPU...{RESET}")

    vectorstore = FAISS.from_documents(chunks, embeddings_model)
    vectorstore.save_local("base_documind")

    print(f"""
{GREEN}{BOLD}
╔══════════════════════════════════════════════╗
║         ✅ BASE CREATED SUCCESSFULLY!          ║
╚══════════════════════════════════════════════╝{RESET}
{GREEN}
   📊 Summary:
   • Pages loaded  : {len(documents)}
   • Chunks created: {len(chunks)}
   • Saved to      : base_documind/

   🚀 Next step: run 'streamlit run app.py'
{RESET}""")


if __name__ == "__main__":
    main()