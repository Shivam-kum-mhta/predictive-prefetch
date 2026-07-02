import { useEffect, useState } from "react";
import { loadNews } from "./newsData";
import "./App.css";

function ArticleSidebar({ article, isOpen, onClose }) {
  if (!article) return null;

  const getCategoryImage = (category) => {
    const images = {
      'sports': 'https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=800&h=400&fit=crop',
      'news': 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=800&h=400&fit=crop',
      'entertainment': 'https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=800&h=400&fit=crop',
      'lifestyle': 'https://images.unsplash.com/photo-1506126613408-eca07ce68773?w=800&h=400&fit=crop',
      'health': 'https://images.unsplash.com/photo-1505751172876-fa1923c5c528?w=800&h=400&fit=crop',
      'finance': 'https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800&h=400&fit=crop',
      'autos': 'https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=800&h=400&fit=crop',
      'tv': 'https://images.unsplash.com/photo-1522869635100-9f4c5e86aa37?w=800&h=400&fit=crop',
      'weather': 'https://images.unsplash.com/photo-1504608524841-42fe6f032b4b?w=800&h=400&fit=crop',
      'video': 'https://images.unsplash.com/photo-1492619375914-88005aa9e8fb?w=800&h=400&fit=crop',
      'travel': 'https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=800&h=400&fit=crop',
      'foodanddrink': 'https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=800&h=400&fit=crop'
    };
    return images[category?.toLowerCase()] || 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=800&h=400&fit=crop';
  };

  return (
    <>
      {/* Overlay */}
      <div 
        className={`sidebar-overlay ${isOpen ? 'active' : ''}`}
        onClick={onClose}
      />
      
      {/* Sidebar */}
      <div className={`article-sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <button className="close-btn" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="sidebar-content">
          <div 
            className="sidebar-image"
            style={{ backgroundImage: `url(${getCategoryImage(article.category)})` }}
          />

          <div className="sidebar-body">
            <div className="sidebar-tags">
              <span className="sidebar-category">{article.category}</span>
              {article.subcategory && <span className="sidebar-subcategory">{article.subcategory}</span>}
            </div>

            <h1 className="sidebar-title">{article.title}</h1>

            <div className="sidebar-meta">
              <span>👁 {article.views?.toLocaleString() || 0} views</span>
              <span>📅 {new Date(article.created_at).toLocaleDateString('en-US', { 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
              })}</span>
            </div>

            <div className="sidebar-abstract">
              <h3>Summary</h3>
              <p>{article.abstract}</p>
            </div>

            <a 
              href={article.url} 
              target="_blank" 
              rel="noopener noreferrer"
              className="read-full-article-btn"
            >
              Read Full Article on Source Website →
            </a>
          </div>
        </div>
      </div>
    </>
  );
}

function NewsCard({ article, onClick, isRecommended }) {
  // Generate a placeholder image based on category
  const getCategoryImage = (category) => {
    const images = {
      'sports': 'https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=400&h=250&fit=crop',
      'news': 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=400&h=250&fit=crop',
      'entertainment': 'https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=400&h=250&fit=crop',
      'lifestyle': 'https://images.unsplash.com/photo-1506126613408-eca07ce68773?w=400&h=250&fit=crop',
      'health': 'https://images.unsplash.com/photo-1505751172876-fa1923c5c528?w=400&h=250&fit=crop',
      'finance': 'https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=400&h=250&fit=crop',
      'autos': 'https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=400&h=250&fit=crop',
      'tv': 'https://images.unsplash.com/photo-1522869635100-9f4c5e86aa37?w=400&h=250&fit=crop',
      'weather': 'https://images.unsplash.com/photo-1504608524841-42fe6f032b4b?w=400&h=250&fit=crop',
      'video': 'https://images.unsplash.com/photo-1492619375914-88005aa9e8fb?w=400&h=250&fit=crop',
      'travel': 'https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=400&h=250&fit=crop',
      'foodanddrink': 'https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=400&h=250&fit=crop'
    };
    return images[category?.toLowerCase()] || 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=400&h=250&fit=crop';
  };

  const handleCardClick = async (e) => {
    e.preventDefault();
    console.log('Article clicked:', article.article_id);
    
    // Track the article click and wait for API call, also open sidebar
    await onClick(article.article_id, article);
  };

  return (
    <article className="news-card">
      {isRecommended && <span className="recommended-badge">⭐ Recommended</span>}
      
      <div className="card-image" style={{ backgroundImage: `url(${getCategoryImage(article.category)})` }}>
        <div className="image-overlay"></div>
      </div>
      
      <div className="card-content">
        <div className="category-tags">
          <span className="category-tag">{article.category}</span>
          {article.subcategory && <span className="subcategory-tag">{article.subcategory}</span>}
        </div>
        
        <h2 className="article-title">{article.title}</h2>
        
        <p className="article-abstract">
          {article.abstract?.length > 120 ? article.abstract.substring(0, 120) + '...' : article.abstract}
        </p>
        
        <div className="card-footer">
          <div className="meta-info">
            <span className="views">👁 {article.views?.toLocaleString() || 0}</span>
            <span className="date">📅 {new Date(article.created_at).toLocaleDateString()}</span>
          </div>
          
          <div className="card-actions">
            <button className="read-more-btn" onClick={handleCardClick}>
              Read More →
            </button>
          </div>
        </div>
      </div>
    </article>
  );
}

function App() {
  const [articles, setArticles] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [userHistory, setUserHistory] = useState([]);
  const [userId] = useState("012345"); // Default user ID
  const [isLoading, setIsLoading] = useState(true);
  const [recommendedArticles, setRecommendedArticles] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedArticle, setSelectedArticle] = useState(null);

  useEffect(() => {
    loadNews().then(data => {
      setArticles(data);
      setIsLoading(false);
    });
  }, []);

  const handleCardClick = async (id, article) => {
    console.log('=== API Call Started ===');
    console.log('Article ID:', id);
    
    // Open sidebar with selected article
    setSelectedArticle(article);
    setSidebarOpen(true);
    
    setSelectedId(id);

    const nextHistory = [...userHistory, id];
    setUserHistory(nextHistory);

    const payload = {
      user_history: nextHistory,
      user_id: userId,
    };

    console.log('Sending payload:', payload);

    try {
      const response = await fetch(
        "http://localhost:9090/model/predict",
        {
          method: "POST",
          headers: {
            accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );

      console.log('Response status:', response.status);

      if (!response.ok) {
        console.error("Prediction request failed:", response.status, response.statusText);
        return;
      }

      const data = await response.json();
      console.log("Prediction response:", data);

      if (data && Array.isArray(data.predictions)) {
        const predictions = data.predictions.slice(0, 10);
        setRecommendedArticles(predictions.map(p => p.article_id));
        setArticles(predictions);
        console.log('Updated articles with', predictions.length, 'recommendations');
      }
      
      console.log('=== API Call Completed ===');
    } catch (error) {
      console.error("Prediction request error:", error);
    }
  };

  const closeSidebar = () => {
    setSidebarOpen(false);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">📰</span>
            <h1>NewsHub</h1>
          </div>
          <div className="user-info">
            <span className="history-count">📚 {userHistory.length} articles read</span>
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="content-wrapper">
          {userHistory.length > 0 && (
            <div className="recommendations-header">
              <h2>✨ Recommended For You</h2>
              <p>Based on your reading history</p>
            </div>
          )}
          
          {isLoading ? (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>Loading news...</p>
            </div>
          ) : (
            <div className="news-grid">
              {articles.map(article => (
                <NewsCard 
                  key={article.article_id} 
                  article={article} 
                  onClick={handleCardClick}
                  isRecommended={recommendedArticles.includes(article.article_id)}
                />
              ))}
            </div>
          )}
        </div>
      </main>

      <footer className="footer">
        <p>© 2025 NewsHub - Powered by AI Recommendations</p>
      </footer>

      {/* Article Sidebar */}
      <ArticleSidebar 
        article={selectedArticle} 
        isOpen={sidebarOpen} 
        onClose={closeSidebar} 
      />
    </div>
  );
}

export default App;