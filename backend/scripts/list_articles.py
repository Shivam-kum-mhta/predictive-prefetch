import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models.markov.markov import SparseMarkovChain
from db.articles import ArticleDatabase

def main():
    model_path = Path(__file__).parent / "models/markov/markov_model.pkl"
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    print("Loading Markov model...")
    model = SparseMarkovChain()
    model.load_model(str(model_path))
    
    print(f"\nModel loaded successfully!")
    print(f"Total articles in model: {model.n_states:,}\n")
    
    print("Connecting to article database...")
    db = ArticleDatabase()
    
    print("First 20 articles (with titles):")
    article_ids = list(model.states)[:20]
    articles = db.get_articles_by_ids(article_ids)
    
    article_map = {a['article_id']: a['title'] for a in articles}
    
    for i, article_id in enumerate(article_ids, 1):
        title = article_map.get(article_id, "No title found")
        title_preview = title[:60] + "..." if len(title) > 60 else title
        print(f"  {i:2d}. {article_id}")
        print(f"      {title_preview}")
    
    print(f"\n... and {model.n_states - 20:,} more articles")
    
    print("\n" + "="*60)
    print("USAGE EXAMPLE:")
    print("="*60)
    if model.states:
        example_article = model.states[0]
        print(f"py visualize_markov.py \"{example_article}\"")
        print(f"py visualize_markov.py \"{example_article}\" --depth 2 --top-k 3")
        print(f"py visualize_markov.py \"{example_article}\" --output my_graph.html")

if __name__ == "__main__":
    main()


