#!/usr/bin/env python3
"""
One-time script to embed all articles from the MIND dataset into ChromaDB
This processes 51,282+ articles with titles and abstracts for semantic search.
"""

from embeddings import NewsEmbeddings
import pandas as pd
import time
import os
from datetime import datetime

def main():
    print("🚀 Starting article embedding process...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load dataset
    dataset_path = '../../dataset/train_news.tsv'
    print(f"📁 Loading dataset from: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path, 
                        header=None, 
                        names=['NewsID', 'Category', 'Subcategory', 'Title', 'Abstract', 
                               'URL', 'TitleEntities', 'AbstractEntities'],
                        sep='\t')
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_path}")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f"📊 Found {len(df):,} articles to embed")
    
    # Show category distribution
    print("\n📈 Category distribution:")
    category_counts = df['Category'].value_counts()
    for category, count in category_counts.head(10).items():
        print(f"  {category}: {count:,} articles")
    if len(category_counts) > 10:
        print(f"  ... and {len(category_counts)-10} more categories")
    
    # Initialize embeddings system
    print(f"\n🔧 Initializing NewsEmbeddings system...")
    try:
        news_embeddings = NewsEmbeddings()
        
        # Check if embeddings already exist and get existing IDs
        stats = news_embeddings.GetCollectionStats()
        existing_ids = set()
        
        if stats['title_count'] > 0 or stats['abstract_count'] > 0:
            print(f"⚠️  Found existing embeddings:")
            print(f"   Titles: {stats['title_count']:,}")
            print(f"   Abstracts: {stats['abstract_count']:,}")
            
            # Get all existing NewsIDs to avoid re-processing
            print("🔍 Checking which articles are already embedded...")
            try:
                all_titles = news_embeddings.title_collection.get()
                existing_ids = set(all_titles['ids'])
                print(f"📋 Found {len(existing_ids):,} already embedded NewsIDs")
            except Exception as e:
                print(f"⚠️ Could not check existing IDs: {e}")
                
            choice = input("Resume from where left off? (Y/n): ").strip().lower()
            if choice == 'n':
                print("🛑 Aborted by user")
                return
        else:
            print("✅ No existing embeddings found - starting fresh")
                
    except Exception as e:
        print(f"❌ Error initializing embeddings: {e}")
        return
    
    # Process articles
    print(f"\n📝 Starting embedding process...")
    start_time = time.time()
    successful_embeds = 0
    failed_embeds = 0
    
    skipped_existing = 0
    
    for index, row in df.iterrows():
        try:
            # Skip if NewsID is missing (only critical field)
            if pd.isna(row['NewsID']):
                failed_embeds += 1
                print(f"   ❌ Skipped article at index {index} - missing NewsID")
                continue
            
            # Skip if already embedded (RESUME FUNCTIONALITY)
            if row['NewsID'] in existing_ids:
                skipped_existing += 1
                if skipped_existing % 1000 == 0:
                    print(f"   ⏭️  Skipped {skipped_existing:,} already embedded articles...")
                continue
            
            # Handle missing title/abstract data gracefully
            title = row['Title'] if not pd.isna(row['Title']) else ""
            abstract = row['Abstract'] if not pd.isna(row['Abstract']) else ""
            
            # Apply fallback logic for missing data
            if not title and not abstract:
                # Both missing - use placeholder
                title = f"[Article {row['NewsID']} - No Title Available]"
                abstract = f"[Article {row['NewsID']} - No Abstract Available]"
                if index < 10:  # Show first few cases for awareness
                    print(f"   📝 Using placeholders for NewsID {row['NewsID']} - both title and abstract missing")
            elif not title:
                # Title missing - use abstract as title
                title = abstract
                if index < 10:
                    print(f"   📝 Using abstract as title for NewsID {row['NewsID']}")
            elif not abstract:
                # Abstract missing - use title as abstract
                abstract = title
                if index < 10:
                    print(f"   📝 Using title as abstract for NewsID {row['NewsID']}")
                
            # Create metadata
            metadata = {
                'category': row['Category'] if not pd.isna(row['Category']) else 'unknown',
                'subcategory': row['Subcategory'] if not pd.isna(row['Subcategory']) else 'unknown',
                'url': row['URL'] if not pd.isna(row['URL']) else None
            }
            
            # Embed title and abstract for this NewsID
            news_embeddings.EmbedTitle([title], [row['NewsID']], [metadata])
            news_embeddings.EmbedAbstract([abstract], [row['NewsID']], [metadata])
            
            successful_embeds += 1
            
            # Enhanced Progress reporting - every 100 articles
            if successful_embeds % 100 == 0:
                elapsed = time.time() - start_time
                rate = successful_embeds / elapsed if elapsed > 0 else 0
                remaining_articles = len(df) - index - 1
                eta_seconds = remaining_articles / rate if rate > 0 else 0
                
                print(f"   ✅ Embedded {successful_embeds:,} articles "
                      f"({index+1:,}/{len(df):,} processed = {((index+1)/len(df)*100):.1f}%) "
                      f"- Rate: {rate:.1f}/sec - ETA: {eta_seconds/60:.1f}min")
                
                # Show current article info
                print(f"      📰 Latest: [{row['Category']}] {row['NewsID']} - {row['Title'][:50]}...")
            
            # Major milestone reporting
            elif successful_embeds % 1000 == 0:
                elapsed = time.time() - start_time
                rate = successful_embeds / elapsed if elapsed > 0 else 0
                print(f"   🎉 MILESTONE: {successful_embeds:,} articles embedded! "
                      f"Average rate: {rate:.1f}/sec - Time elapsed: {elapsed/60:.1f}min")
            
        except Exception as e:
            failed_embeds += 1
            print(f"   ❌ Error embedding NewsID {row.get('NewsID', 'unknown')} at index {index}: {e}")
            if failed_embeds < 5:  # Show first few errors for debugging
                print(f"      Title: {row.get('Title', 'N/A')[:50]}...")
                print(f"      Category: {row.get('Category', 'N/A')}")
    
    # Final statistics
    total_time = time.time() - start_time
    final_stats = news_embeddings.GetCollectionStats()
    
    print(f"\n✅ Embedding process completed!")
    print(f"⏰ Total time: {total_time/60:.1f} minutes")
    print(f"📊 Results:")
    print(f"   ✅ Newly embedded: {successful_embeds:,} articles")
    print(f"   ⏭️  Skipped (already embedded): {skipped_existing:,} articles")
    print(f"   ❌ Failed: {failed_embeds:,} articles")
    print(f"   📈 New embedding rate: {successful_embeds/total_time:.1f} articles/second")
    print(f"\n📚 Final collection stats:")
    print(f"   📖 Total titles: {final_stats['title_count']:,}")
    print(f"   📄 Total abstracts: {final_stats['abstract_count']:,}")
    
    # Database location
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    print(f"💾 Database location: {db_path}")

if __name__ == "__main__":
    main()