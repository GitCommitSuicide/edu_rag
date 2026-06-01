"""
Console App — RAG with LlamaIndex + Ollama
Run: python app.py
"""

import sys, os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag_engine import configure_settings, get_index, query, RAGConfig

def main():
    # Initialize config first to get defaults
    cfg = RAGConfig()
    
    parser = argparse.ArgumentParser(description="RAG Assistant Console App")
    parser.add_argument("--llm", type=str, default=cfg.LLM_MODEL, help=f"Ollama LLM Model (default: {cfg.LLM_MODEL})")
    parser.add_argument("--embed", type=str, default=cfg.EMBED_MODEL, help=f"Embedding Model (default: {cfg.EMBED_MODEL})")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index")
    
    args = parser.parse_args()
    
    cfg.LLM_MODEL = args.llm
    cfg.EMBED_MODEL = args.embed
    
    if args.rebuild:
        import shutil
        print("[App] Rebuilding index: removing existing vectorstore...")
        shutil.rmtree(cfg.CHROMA_DB_DIR, ignore_errors=True)
    
    print(f"[Settings] LLM={cfg.LLM_MODEL}, Embed={cfg.EMBED_MODEL}")
    configure_settings(cfg)
    
    try:
        index = get_index(cfg)
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print("Add files to the `data/` folder and run again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] Failed to load index: {e}")
        print("Make sure Ollama is running (`ollama serve`) and the model is pulled.")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("RAG Assistant Ready")
    print("Type your question below (or type 'exit' / 'quit' to stop).")
    print("="*50 + "\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ['exit', 'quit']:
                print("Bye!")
                break
            if not prompt:
                continue
                
            print("Thinking...")
            result = query(prompt, index, cfg)
            print(f"\n[Answer]\n{result['answer']}\n")
            
            if result['sources']:
                print("--- Sources ---")
                for s in result['sources']:
                    print(f"- {s['file']} (score: {s['score']})")
            print()
            
        except KeyboardInterrupt:
            print("\nBye!")
            break

if __name__ == "__main__":
    main()
