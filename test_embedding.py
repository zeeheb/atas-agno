import logging
import os
import traceback
from dotenv import load_dotenv, find_dotenv
from agno.embedder.openai import OpenAIEmbedder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import the SimpleVectorStore class
from main import SimpleVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_test_case(description, test_func):
    """Run a test case with proper error handling and reporting"""
    print(f"\n{'='*20} {description} {'='*20}")
    try:
        test_func()
        print(f"✅ {description} - PASSED")
        return True
    except Exception as e:
        print(f"❌ {description} - FAILED: {str(e)}")
        print(traceback.format_exc())
        return False

def main():
    # Load environment variables
    _ = load_dotenv(find_dotenv())
    
    # Track test success/failure
    tests_total = 0
    tests_passed = 0
    
    try:
        # Initialize the embedder
        embedder = OpenAIEmbedder()
        
        # Test embedder
        tests_total += 1
        def test_embedding():
            test_text = "This is a test document for embedding."
            embedding = embedder.get_embedding(test_text)
            print(f"Successfully generated embedding with length: {len(embedding)}")
            assert len(embedding) > 0, "Embedding should have a positive length"
            
        if run_test_case("OpenAI Embedder", test_embedding):
            tests_passed += 1
        
        # Initialize vector store
        vector_store = SimpleVectorStore(embedder)
        
        # Test adding documents
        tests_total += 1
        def test_add_docs():
            # Add documents to vector store
            docs = [
                "Document 1: This is about artificial intelligence.",
                "Document 2: This is about machine learning.",
                "Document 3: This is about natural language processing."
            ]
            
            success = vector_store.add(texts=docs)
            assert success, "Adding documents should return True"
            assert len(vector_store) == 3, f"Expected 3 documents, got {len(vector_store)}"
            print(f"Successfully added {len(docs)} documents to the vector store.")
            
        if run_test_case("Adding Documents", test_add_docs):
            tests_passed += 1
        
        # Test similarity search
        tests_total += 1
        def test_similarity_search():
            query = "Tell me about AI"
            results = vector_store.similarity_search(query, k=2)
            
            assert len(results) == 2, f"Expected 2 results, got {len(results)}"
            
            print(f"\nSearch Query: '{query}'")
            print(f"Top {len(results)} results:")
            for i, result in enumerate(results):
                print(f"{i+1}. {result['text']} (Score: {result['score']:.4f})")
                assert result['score'] >= 0 and result['score'] <= 1, f"Score should be between 0 and 1, got {result['score']}"
                
        if run_test_case("Similarity Search", test_similarity_search):
            tests_passed += 1
            
        # Test vector store stats
        tests_total += 1
        def test_stats():
            stats = vector_store.stats()
            print(f"Vector store stats: {stats}")
            assert 'document_count' in stats, "Stats should include document count"
            assert stats['document_count'] > 0, "Document count should be positive"
            
        if run_test_case("Vector Store Stats", test_stats):
            tests_passed += 1
            
        # Test edge cases
        tests_total += 1
        def test_edge_cases():
            # Empty query
            empty_results = vector_store.similarity_search("", k=1)
            print(f"Empty query results: {len(empty_results)}")
            
            # None k value
            none_k_results = vector_store.similarity_search("AI", k=None)
            print(f"None k results: {len(none_k_results)}")
            
            # Very large k value
            large_k_results = vector_store.similarity_search("AI", k=100)
            print(f"Large k results: {len(large_k_results)}")
            assert len(large_k_results) <= len(vector_store), "Results should not exceed document count"
            
        if run_test_case("Edge Cases", test_edge_cases):
            tests_passed += 1
            
    except Exception as e:
        logger.error(f"Error in test suite: {str(e)}")
        print(f"Test suite error: {str(e)}")
        print(traceback.format_exc())
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary: {tests_passed}/{tests_total} tests passed")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 