import logging
from main import SimpleVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockEmbedder:
    """Mock embedder for testing"""
    def get_embedding(self, text):
        # Return a simple embedding of length 5
        return [0.1, 0.2, 0.3, 0.4, 0.5]

def main():
    try:
        # Initialize with mock embedder (no API keys needed)
        mock_embedder = MockEmbedder()
        vector_store = SimpleVectorStore(mock_embedder)
        
        # Add sample documents
        docs = [
            "Document 1: This is about artificial intelligence.",
            "Document 2: This is about machine learning."
        ]
        
        # Add to vector store
        vector_store.add(texts=docs)
        print(f"Added {len(docs)} documents to the vector store.")
        
        # Test with None k value
        query = "Tell me about AI"
        
        # Test with a None k value - this would previously fail
        print("\nTesting similarity_search with k=None:")
        try:
            results = vector_store.similarity_search(query, k=None)
            print(f"Success! Got {len(results)} results")
            for i, result in enumerate(results):
                print(f"{i+1}. {result['text']}")
        except Exception as e:
            print(f"Error: {str(e)}")
        
        # Test search method with num_documents=None
        print("\nTesting search method with num_documents=None:")
        try:
            documents = vector_store.search(query, num_documents=None)
            print(f"Success! Got {len(documents)} documents")
            for i, doc in enumerate(documents):
                print(f"{i+1}. {doc.content}")
                
                # Test to_dict method
                doc_dict = doc.to_dict()
                print(f"   - Dictionary conversion: {doc_dict['content'][:30]}...")
        except Exception as e:
            print(f"Error: {str(e)}")
        
        # Test other edge cases
        print("\nTesting with negative k value:")
        results = vector_store.similarity_search(query, k=-5)
        print(f"Got {len(results)} results")
        
        print("\nTesting with zero k value:")
        results = vector_store.similarity_search(query, k=0)
        print(f"Got {len(results)} results")
        
        print("\nTesting with string k value:")
        try:
            results = vector_store.similarity_search(query, k="string")
            print(f"Got {len(results)} results")
        except Exception as e:
            print(f"Error: {str(e)}")
            
        print("\nTest completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 