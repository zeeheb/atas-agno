import logging
import traceback
from dotenv import load_dotenv, find_dotenv
from agno.embedder.openai import OpenAIEmbedder

# Import the SimpleVectorStore from the correct location
from main import SimpleVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    _ = load_dotenv(find_dotenv())
    
    try:
        # Initialize the embedder
        embedder = OpenAIEmbedder()
        
        # Initialize vector store
        vector_store = SimpleVectorStore(embedder)
        
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
        
        # Test with a None k value
        print("\nTesting with k=None:")
        try:
            results = vector_store.similarity_search(query, k=None)
            print(f"Success! Got {len(results)} results")
            for i, result in enumerate(results):
                print(f"{i+1}. {result['text']}")
        except Exception as e:
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
        
        # Also test the search method for agno compatibility
        print("\nTesting search method with num_documents=None:")
        try:
            documents = vector_store.search(query, num_documents=None)
            print(f"Success! Got {len(documents)} documents")
            for i, doc in enumerate(documents):
                print(f"{i+1}. {doc['content']}")
        except Exception as e:
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
        
        print("\nTest completed!")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()