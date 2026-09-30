import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── ANSI Colors ────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ── Test Questions ─────────────────────────────────────────────
# Format: (question, expected_keywords, topic)
QUESTIONS_TEST = [
    # Linux questions
    (
        "How do I list files in a directory in Linux?",
        ["ls", "directory", "list"],
        "Linux"
    ),
    (
        "How to change file permissions in Linux?",
        ["chmod", "permission", "read", "write"],
        "Linux"
    ),
    (
        "How to find files using the command line?",
        ["find", "search", "locate"],
        "Linux"
    ),
    (
        "How to manage processes in Linux?",
        ["ps", "kill", "process", "top"],
        "Linux"
    ),
    # Cisco / Network questions
    (
        "How to configure a static route on Cisco router?",
        ["ip route", "static", "gateway", "router"],
        "Cisco"
    ),
    (
        "How to configure VLAN on a Cisco switch?",
        ["vlan", "switchport", "access", "trunk"],
        "Cisco"
    ),
    (
        "How to enable SSH on Cisco device?",
        ["ssh", "crypto", "hostname", "domain"],
        "Cisco"
    ),
    (
        "What is OSPF and how to configure it?",
        ["ospf", "router", "network", "area"],
        "Cisco"
    ),
]

SCORE_THRESHOLD = 1.5


def initialiser():
    """Initialize system components."""
    print(f"\n{BOLD}🔄 Initializing system...{RESET}")

    if not os.path.exists("base_documind"):
        print(f"{RED}❌ Base 'base_documind' not found. Run main.py first.{RESET}")
        exit(1)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.load_local(
        "base_documind",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512  # Short answers for evaluation
    )

    print(f"{GREEN}✅ System ready{RESET}\n")
    return vectorstore, llm


def evaluer_retrieval(vectorstore, question: str, mots_cles: list) -> dict:
    """Evaluate the retrieval quality for one question."""

    debut = time.time()
    resultats = vectorstore.similarity_search_with_score(question, k=3)
    temps = time.time() - debut

    if not resultats:
        return {
            "retrieved": False,
            "best_score": None,
            "above_threshold": False,
            "chunks_found": 0,
            "sources": [],
            "retrieval_time": temps,
            "keyword_hits": 0,
        }

    best_score = resultats[0][1]

    # Check how many results pass the threshold
    chunks_above = [
        (doc, score) for doc, score in resultats
        if score < SCORE_THRESHOLD
    ]

    # Check keyword presence in retrieved chunks
    all_text = " ".join([
        doc.page_content.lower() for doc, _ in resultats
    ])
    keyword_hits = sum(
        1 for kw in mots_cles if kw.lower() in all_text
    )

    sources = list(set([
        f"{doc.metadata.get('source', 'Unknown')} p.{doc.metadata.get('page', '?')}"
        for doc, _ in resultats
    ]))

    return {
        "retrieved": True,
        "best_score": round(best_score, 4),
        "above_threshold": best_score < SCORE_THRESHOLD,
        "chunks_found": len(chunks_above),
        "sources": sources,
        "retrieval_time": round(temps, 3),
        "keyword_hits": keyword_hits,
        "total_keywords": len(mots_cles),
    }


def evaluer_generation(llm, vectorstore, question: str) -> dict:
    """Evaluate the generation quality for one question."""

    # Get context
    resultats = vectorstore.similarity_search_with_score(question, k=3)
    chunks_pertinents = [
        (doc, score) for doc, score in resultats
        if score < SCORE_THRESHOLD
    ]

    if not chunks_pertinents:
        return {
            "generated": False,
            "response": "No relevant context found.",
            "gen_time": 0,
            "length": 0,
        }

    contexte = "\n\n".join([doc.page_content for doc, _ in chunks_pertinents])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical assistant for Linux and Cisco.
Answer ONLY using the context below. Be concise (max 3 sentences).
If not in context, say "Not found in documentation."

Context: {contexte}"""),
        ("human", "{question}")
    ])

    chain = prompt | llm

    debut = time.time()
    try:
        reponse = chain.invoke({
            "contexte": contexte,
            "question": question
        })
        temps = time.time() - debut
        reponse_text = reponse.content
        generated = True
    except Exception as e:
        reponse_text = f"Error: {e}"
        temps = 0
        generated = False

    return {
        "generated": generated,
        "response": reponse_text,
        "gen_time": round(temps, 3),
        "length": len(reponse_text.split()),
    }


def afficher_barre(valeur: float, max_val: float = 1.0, largeur: int = 20) -> str:
    """Display a simple ASCII progress bar."""
    ratio = min(valeur / max_val, 1.0)
    filled = int(ratio * largeur)
    bar = "█" * filled + "░" * (largeur - filled)
    pct = ratio * 100
    return f"[{bar}] {pct:.0f}%"


def main():
    print(f"""
{BLUE}{BOLD}
╔══════════════════════════════════════════════╗
║      📊 DocuMind - RAG Evaluation Report      ║
║        Linux & Cisco Knowledge Base          ║
╚══════════════════════════════════════════════╝
{RESET}""")

    vectorstore, llm = initialiser()

    resultats_globaux = []

    # ── Run Evaluation ─────────────────────────────────────────
    print(f"{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}🔍 Running evaluation on {len(QUESTIONS_TEST)} test questions...{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    for i, (question, mots_cles, topic) in enumerate(QUESTIONS_TEST, 1):

        topic_icon = "🐧" if topic == "Linux" else "🔧"
        print(f"{CYAN}{BOLD}[{i}/{len(QUESTIONS_TEST)}] {topic_icon} {topic}{RESET}")
        print(f"  ❓ {question}")

        # Retrieval evaluation
        ret = evaluer_retrieval(vectorstore, question, mots_cles)

        # Generation evaluation
        gen = evaluer_generation(llm, vectorstore, question)

        # Results
        score_color = GREEN if ret.get("above_threshold") else RED
        kw_ratio = ret.get("keyword_hits", 0) / max(ret.get("total_keywords", 1), 1)

        print(f"  ┌─ Retrieval:")
        print(f"  │  • Best score  : {score_color}{ret.get('best_score', 'N/A')}{RESET} "
              f"(threshold < {SCORE_THRESHOLD})")
        print(f"  │  • Relevant chunks : {ret.get('chunks_found', 0)}/3")
        print(f"  │  • Keyword hits    : {ret.get('keyword_hits', 0)}/{ret.get('total_keywords', 0)} "
              f"{afficher_barre(kw_ratio)}")
        print(f"  │  • Retrieval time  : {ret.get('retrieval_time', 0)}s")
        print(f"  │  • Sources:")
        for src in ret.get("sources", []):
            print(f"  │      📄 {src}")
        print(f"  └─ Generation:")
        print(f"     • Time     : {gen.get('gen_time', 0)}s")
        print(f"     • Words    : {gen.get('length', 0)}")
        print(f"     • Preview  : {gen.get('response', '')[:120]}...")
        print()

        resultats_globaux.append({
            "question": question,
            "topic": topic,
            "retrieval": ret,
            "generation": gen,
        })

        # Small pause to respect API rate limits
        time.sleep(1)

    # ── Global Statistics ──────────────────────────────────────
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}📊 GLOBAL EVALUATION SUMMARY{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    total = len(resultats_globaux)

    retrieved_ok = sum(
        1 for r in resultats_globaux
        if r["retrieval"].get("above_threshold")
    )

    avg_score = sum(
        r["retrieval"].get("best_score", 0) or 0
        for r in resultats_globaux
    ) / total

    avg_ret_time = sum(
        r["retrieval"].get("retrieval_time", 0)
        for r in resultats_globaux
    ) / total

    avg_gen_time = sum(
        r["generation"].get("gen_time", 0)
        for r in resultats_globaux
    ) / total

    avg_kw = sum(
        r["retrieval"].get("keyword_hits", 0) /
        max(r["retrieval"].get("total_keywords", 1), 1)
        for r in resultats_globaux
    ) / total

    generated_ok = sum(
        1 for r in resultats_globaux
        if r["generation"].get("generated")
    )

    # By topic
    linux_ok = sum(
        1 for r in resultats_globaux
        if r["topic"] == "Linux" and r["retrieval"].get("above_threshold")
    )
    cisco_ok = sum(
        1 for r in resultats_globaux
        if r["topic"] == "Cisco" and r["retrieval"].get("above_threshold")
    )
    linux_total = sum(1 for r in resultats_globaux if r["topic"] == "Linux")
    cisco_total = sum(1 for r in resultats_globaux if r["topic"] == "Cisco")

    retrieval_rate = retrieved_ok / total
    gen_rate = generated_ok / total

    print(f"  {'Metric':<35} {'Value':<15} {'Visual'}")
    print(f"  {'─' * 70}")
    print(f"  {'✅ Retrieval success rate':<35} {retrieved_ok}/{total:<13} "
          f"{afficher_barre(retrieval_rate)}")
    print(f"  {'🤖 Generation success rate':<35} {generated_ok}/{total:<13} "
          f"{afficher_barre(gen_rate)}")
    # ═══════════════════════════════════════════════════════════
    # FIXED LINE BELOW: was .1%:<13, now <13.1%
    # ═══════════════════════════════════════════════════════════
    print(f"  {'🔑 Avg keyword match rate':<35} {avg_kw:<13.1%} "
          f"{afficher_barre(avg_kw)}")
    print(f"  {'📊 Avg similarity score':<35} {avg_score:.4f}")
    print(f"  {'⚡ Avg retrieval time':<35} {avg_ret_time:.3f}s")
    print(f"  {'🤖 Avg generation time':<35} {avg_gen_time:.3f}s")
    print(f"  {'🐧 Linux success rate':<35} {linux_ok}/{linux_total:<13} "
          f"{afficher_barre(linux_ok / max(linux_total, 1))}")
    print(f"  {'🔧 Cisco success rate':<35} {cisco_ok}/{cisco_total:<13} "
          f"{afficher_barre(cisco_ok / max(cisco_total, 1))}")

    # ── Overall Grade ──────────────────────────────────────────
    overall_score = (retrieval_rate * 0.5 + avg_kw * 0.3 + gen_rate * 0.2) * 100

    if overall_score >= 80:
        grade_color = GREEN
        grade = "🏆 EXCELLENT"
    elif overall_score >= 60:
        grade_color = YELLOW
        grade = "✅ GOOD"
    else:
        grade_color = RED
        grade = "⚠️  NEEDS IMPROVEMENT"

    print(f"""
{BOLD}{'═' * 60}{RESET}
{grade_color}{BOLD}  🎯 OVERALL RAG SCORE: {overall_score:.1f}/100 → {grade}{RESET}
{BOLD}{'═' * 60}{RESET}

  Score breakdown:
    • Retrieval success  (50%): {retrieval_rate * 100:.1f}/100
    • Keyword match      (30%): {avg_kw * 100:.1f}/100
    • Generation success (20%): {gen_rate * 100:.1f}/100

{YELLOW}  💡 Tips to improve score:
    • Add more PDF documents with ajouter_document.py
    • Adjust chunk_size in main.py (try 800 or 1200)
    • Lower the score threshold if too strict{RESET}
""")


if __name__ == "__main__":
    main()