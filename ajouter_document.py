import sys
import os
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


def print_usage():
    print(f"""
{BOLD}Usage:{RESET}
    python ajouter_document.py <path_to_pdf>

{BOLD}Examples:{RESET}
    python ajouter_document.py data/nouveau_manuel.pdf
    python ajouter_document.py data/cisco_guide.pdf
    """)


def main():
    print(f"\n{BLUE}{BOLD}📄 DocuMind - Add Document{RESET}\n")

    if len(sys.argv) < 2:
        print(f"{RED}❌ No file specified.{RESET}")
        print_usage()
        sys.exit(1)

    fichier = sys.argv[1]

    if not os.path.exists(fichier):
        print(f"{RED}❌ File not found: '{fichier}'{RESET}")
        print(f"{YELLOW}👉 Check the path and try again.{RESET}")
        sys.exit(1)

    if not fichier.lower().endswith(".pdf"):
        print(f"{RED}❌ Only PDF files are supported.{RESET}")
        print(f"{YELLOW}👉 Provided file: '{fichier}'{RESET}")
        sys.exit(1)

    if not os.path.exists("base_documind"):
        print(f"{RED}❌ Vector base 'base_documind' not found.{RESET}")
        print(f"{YELLOW}👉 Run main.py first to create the base.{RESET}")
        sys.exit(1)

    size_mb = os.path.getsize(fichier) / (1024 * 1024)
    nom_fichier = os.path.basename(fichier)
    print(f"   📄 File  : {nom_fichier}")
    print(f"   💾 Size  : {size_mb:.1f} MB")
    print()

    print(f"{BOLD}🧠 STEP 1/4 - Loading embedding model...{RESET}")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"{GREEN}✅ Model loaded{RESET}\n")

    print(f"{BOLD}📂 STEP 2/4 - Loading existing vector base...{RESET}")
    vectorstore = FAISS.load_local(
        "base_documind",
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    print(f"{GREEN}✅ Existing base loaded{RESET}\n")

    print(f"{BOLD}📖 STEP 3/4 - Loading new PDF...{RESET}")
    try:
        loader = PyPDFLoader(fichier)
        documents = loader.load()
        # Add clean source metadata
        for doc in documents:
            doc.metadata["source"] = nom_fichier
        print(f"   📃 Pages loaded: {len(documents)}")
        print(f"{GREEN}✅ PDF loaded{RESET}\n")
    except Exception as e:
        print(f"{RED}❌ Error reading PDF: {e}{RESET}")
        sys.exit(1)

    print(f"{BOLD}✂️  STEP 4/4 - Splitting and indexing...{RESET}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=[
            "\n## ", "\n### ", "\n#### ",
            "\n\n", "\n", ". ", " ", ""
        ]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   📦 Chunks created: {len(chunks)}")
    print(f"{YELLOW}⏳ Vectorizing... please wait...{RESET}")

    vectorstore.add_documents(chunks)
    vectorstore.save_local("base_documind")

    print(f"""
{GREEN}{BOLD}
╔══════════════════════════════════════════════╗
║       ✅ DOCUMENT ADDED SUCCESSFULLY!          ║
╚══════════════════════════════════════════════╝{RESET}
{GREEN}
   📄 File   : {nom_fichier}
   📃 Pages  : {len(documents)}
   📦 Chunks : {len(chunks)}
   💾 Saved to: base_documind/

   🚀 The new document is now searchable!
{RESET}""")


if __name__ == "__main__":
    main()