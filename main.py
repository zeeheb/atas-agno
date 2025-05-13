import sys
import os
import gc
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                            QFileDialog, QMessageBox, QProgressDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv, find_dotenv
from agno.embedder.openai import OpenAIEmbedder
import logging
import psutil
import time
from PyPDF2 import PdfReader
import json
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class SimpleVectorStore:
    """Simple vector store implementation using numpy and cosine similarity"""
    def __init__(self, embedder):
        self.embedder = embedder
        self.documents = []
        self.vectors = []
        self.metadatas = []
        self.ids = []

    def add(self, texts, metadatas=None, ids=None):
        """Add documents to the vector store"""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Get embeddings for the texts using get_embedding method
        embeddings = []
        for text in texts:
            embeddings.append(self.embedder.get_embedding(text))
        
        # Add to store
        self.documents.extend(texts)
        self.vectors.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def search(self, query, num_documents=5, **kwargs):
        """Search method compatible with agno's expected interface"""
        # Handle None or invalid num_documents parameter
        if num_documents is None or not isinstance(num_documents, int) or num_documents <= 0:
            logger.info(f"Invalid num_documents value: {num_documents}, using default of 5")
            num_documents = 5
            
        results = self.similarity_search(query, k=num_documents)
        
        # Convert to Document objects for agno compatibility
        documents = []
        for result in results:
            # Create a Document object with a to_dict method
            doc = AgnoCompatibleDocument(
                content=result["text"],
                metadata=result["metadata"],
                id=result["id"],
                score=result["score"]
            )
            documents.append(doc)
        
        return documents

    def similarity_search(self, query, k=5):
        """Search for similar documents"""
        if not self.documents:
            return []

        # Handle None k value
        if k is None or not isinstance(k, int) or k <= 0:
            logger.info(f"Invalid k value: {k}, using default of 5")
            k = 5

        # Get query embedding using get_embedding method
        query_embedding = self.embedder.get_embedding(query)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.vectors)[0]
        
        # Ensure we don't exceed the number of available documents
        k = min(k, len(self.documents))
        if k == 0:
            return []
            
        # Get top k results
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            results.append({
                'text': self.documents[idx],
                'metadata': self.metadatas[idx],
                'id': self.ids[idx],
                'score': float(similarities[idx])
            })
        
        return results

class WorkerThread(QThread):
    """Worker thread for processing AI responses"""
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, agent, question):
        super().__init__()
        self.agent = agent
        self.question = question

    def run(self):
        try:
            response = self.agent.print_response(self.question, markdown=True)
            self.response_ready.emit(response)
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            self.error_occurred.emit(str(e))

class DocumentLoaderThread(QThread):
    """Worker thread for loading documents"""
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    file_processed = pyqtSignal(str)
    memory_warning = pyqtSignal(str)

    def __init__(self, vector_db):
        super().__init__()
        self.vector_db = vector_db
        self.is_running = True
        self.memory_threshold = 1000  # MB

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            raise

    def process_text(self, text, chunk_size=1000):
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def run(self):
        try:
            # Force garbage collection before loading
            gc.collect()
            
            # Get list of PDF files
            docs_path = Path('docs')
            pdf_files = list(docs_path.glob('*.pdf'))
            
            if not pdf_files:
                self.error.emit("Nenhum arquivo PDF encontrado no diretório 'docs'")
                return

            # Process each file individually
            for i, pdf_file in enumerate(pdf_files):
                if not self.is_running:
                    break
                    
                try:
                    # Check memory usage
                    memory_usage = get_memory_usage()
                    if memory_usage > self.memory_threshold:
                        self.memory_warning.emit(f"Alto uso de memória: {memory_usage:.1f}MB")
                        gc.collect()
                        time.sleep(1)

                    logger.info(f"Processing file: {pdf_file.name}")
                    self.file_processed.emit(f"Processando: {pdf_file.name}")

                    # Extract text from PDF
                    text = self.extract_text_from_pdf(pdf_file)
                    
                    # Process text into chunks
                    chunks = self.process_text(text)
                    
                    # Store chunks in vector database
                    for chunk in chunks:
                        # Generate a unique ID for each chunk
                        chunk_id = str(uuid.uuid4())
                        
                        # Create metadata
                        metadata = {
                            "source": pdf_file.name,
                            "chunk_id": chunk_id
                        }
                        
                        # Add to vector database
                        self.vector_db.add(
                            texts=[chunk],
                            metadatas=[metadata],
                            ids=[chunk_id]
                        )
                    
                    # Update progress
                    progress = int((i + 1) / len(pdf_files) * 100)
                    self.progress.emit(progress)
                    
                    # Force garbage collection after each file
                    gc.collect()
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing file {pdf_file.name}: {str(e)}")
                    self.error.emit(f"Erro ao processar {pdf_file.name}: {str(e)}")
                    continue

            if self.is_running:
                self.finished.emit()
                
        except Exception as e:
            logger.error(f"Error in document loading: {str(e)}")
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ATAS - Analisador de Atas")
        self.setMinimumSize(800, 600)
        
        # Initialize the AI components
        self._init_ai_components()
        
        # Create the main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create the UI components
        self._create_ui_components(layout)
        
        # Set window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QLabel {
                font-weight: bold;
            }
            QProgressDialog {
                background-color: white;
            }
        """)

    def _init_ai_components(self):
        """Initialize AI components"""
        try:
            _ = load_dotenv(find_dotenv())
            
            # Initialize the embedder
            embedder = OpenAIEmbedder()
            
            # Initialize the vector store
            self.vector_db = SimpleVectorStore(embedder)

            self.agent = Agent(
                model=OpenAIChat(id='gpt-4o-mini'),
                knowledge=self.vector_db,
                search_knowledge=True,
                show_tool_calls=True,
                markdown=True,
                instructions='''
                    "Você é um assistente especializado em analisar atas de reuniões em formato .pdf. Sempre que o usuário fizer uma pergunta,
                    primeiro reformule-a para torná-la mais clara e específica, com foco em palavras-chave, datas, pessoas ou eventos relevantes,
                    para otimizar a busca nos documentos. Após a reformulação, busque nas atas e forneça uma resposta precisa e objetiva, 
                    e se necessário, explique o raciocínio por trás da reformulação antes de realizar a busca. Faça buscas adicionais para garantir que todas as partes da questão sejam abordadas. Responda apenas o que tiver relacionado a pergunta, sem informações extras desnecessárias '''
            )
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            QMessageBox.critical(self, "Erro de Inicialização", 
                               f"Erro ao inicializar componentes: {str(e)}\n"
                               "Verifique se o arquivo .env está configurado corretamente.")

    def _create_ui_components(self, layout):
        """Create and arrange UI components"""
        # Question input
        question_label = QLabel("Faça sua pergunta sobre as atas:")
        layout.addWidget(question_label)
        
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("Digite sua pergunta aqui...")
        self.question_input.setMaximumHeight(100)
        layout.addWidget(self.question_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ask_button = QPushButton("Perguntar")
        self.ask_button.clicked.connect(self.process_question)
        button_layout.addWidget(self.ask_button)
        
        self.load_docs_button = QPushButton("Carregar Documentos")
        self.load_docs_button.clicked.connect(self.load_documents)
        button_layout.addWidget(self.load_docs_button)
        
        layout.addLayout(button_layout)
        
        # Response area
        response_label = QLabel("Resposta:")
        layout.addWidget(response_label)
        
        self.response_output = QTextEdit()
        self.response_output.setReadOnly(True)
        layout.addWidget(self.response_output)

    def process_question(self):
        """Process the user's question"""
        question = self.question_input.toPlainText().strip()
        if not question:
            QMessageBox.warning(self, "Aviso", "Por favor, digite uma pergunta.")
            return
        
        self.ask_button.setEnabled(False)
        self.response_output.clear()
        self.response_output.append("Processando sua pergunta...")
        
        # Create and start worker thread
        self.worker = WorkerThread(self.agent, question)
        self.worker.response_ready.connect(self.handle_response)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(lambda: self.ask_button.setEnabled(True))
        self.worker.start()

    def handle_response(self, response):
        """Handle the AI response"""
        self.response_output.clear()
        self.response_output.append(response)

    def handle_error(self, error_message):
        """Handle any errors that occur"""
        self.response_output.clear()
        self.response_output.append(f"Erro: {error_message}")

    def load_documents(self):
        """Load PDF documents"""
        try:
            # Disable buttons during loading
            self.load_docs_button.setEnabled(False)
            self.ask_button.setEnabled(False)
            
            # Create progress dialog
            self.progress = QProgressDialog("Carregando documentos...", "Cancelar", 0, 100, self)
            self.progress.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress.setMinimumDuration(0)
            self.progress.setAutoClose(False)
            self.progress.setAutoReset(False)
            self.progress.canceled.connect(self.cancel_loading)
            self.progress.show()
            
            # Create and start document loader thread
            self.doc_loader = DocumentLoaderThread(self.vector_db)
            self.doc_loader.progress.connect(self.progress.setValue)
            self.doc_loader.file_processed.connect(self.progress.setLabelText)
            self.doc_loader.memory_warning.connect(self._show_memory_warning)
            self.doc_loader.finished.connect(self._on_documents_loaded)
            self.doc_loader.error.connect(self._on_document_load_error)
            self.doc_loader.start()
            
        except Exception as e:
            logger.error(f"Error starting document loading: {str(e)}")
            self._on_document_load_error(str(e), None)

    def _show_memory_warning(self, message):
        """Show memory warning message"""
        self.progress.setLabelText(message)

    def cancel_loading(self):
        """Cancel the document loading process"""
        if hasattr(self, 'doc_loader'):
            self.doc_loader.stop()
            self.progress.close()
            self.load_docs_button.setEnabled(True)
            self.ask_button.setEnabled(True)

    def _on_documents_loaded(self):
        """Handle successful document loading"""
        self.progress.close()
        self.load_docs_button.setEnabled(True)
        self.ask_button.setEnabled(True)
        QMessageBox.information(self, "Sucesso", "Documentos carregados com sucesso!")

    def _on_document_load_error(self, error_message, progress=None):
        """Handle document loading errors"""
        if progress:
            progress.close()
        self.load_docs_button.setEnabled(True)
        self.ask_button.setEnabled(True)
        QMessageBox.critical(self, "Erro", f"Erro ao carregar documentos: {error_message}")

class AgnoCompatibleDocument:
    """Document class compatible with agno framework's expected interface"""
    def __init__(self, content, metadata=None, id=None, score=None):
        self.content = content
        self.metadata = metadata or {}
        self.id = id
        self.score = score
        
    def to_dict(self):
        """Convert to dictionary, which is required by agno"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "id": self.id,
            "score": self.score
        }
        
    def __repr__(self):
        """String representation of the Document"""
        return f"Document(content={self.content[:50]}..., metadata={self.metadata})"

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
