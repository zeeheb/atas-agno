import sys
import os
import gc
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                            QFileDialog, QMessageBox, QProgressDialog,
                            QStatusBar, QSplashScreen, QScrollArea, QFrame,
                            QStackedWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime, QSize
from PyQt6.QtGui import QFont, QIcon, QPixmap, QColor, QPainter
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
import re

# Import our persistence module
from persistence import VectorStorePersistence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application constants
APP_TITLE = "iATAS - Analisador de ATAS"
APP_VERSION = "1.2.0"
APP_DATA_DIR = "app_data"

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
        
    @staticmethod
    def _clean_ansi_and_formatting(text):
        """Remove ANSI color codes, box drawing characters and other formatting artifacts"""
        import re
        
        if not text:
            return ""
            
        # Remove ANSI escape sequences (color codes)
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        result = ansi_escape.sub('', text)
        
        # Remove box drawing and formatting characters (pipes, etc.)
        result = re.sub(r'[│┌┐└┘┐┘─┬┴┼┤├]', '', result)
        
        # Remove any remaining formatting artifacts
        result = re.sub(r'\[3[0-9]m', '', result)  # Color markers like [34m
        result = re.sub(r'\[0m', '', result)       # Reset markers
        
        # Further cleanup of any artifacts
        return result.strip()

    def run(self):
        try:
            # Use the agent interface but capture the response directly
            # The print_response method may be printing to console but not returning the value

            # First, search for relevant context
            if hasattr(self.agent, 'knowledge') and self.agent.search_knowledge:
                logger.info("Searching knowledge base for relevant context")
                docs = self.agent.knowledge.search(self.question)
                context_texts = [d.content for d in docs]
                context = "\n\n".join(context_texts)
                logger.info(f"Found {len(docs)} relevant documents")
            else:
                context = ""
                
            # Use the complete_chat method directly 
            if hasattr(self.agent.model, 'complete_chat'):
                # For OpenAI models, build the messages
                if context:
                    messages = [
                        {"role": "system", "content": self.agent.instructions + "\nIMPORTANT: Provide ONLY your final answer. Do NOT include your thinking process, reasoning, or intermediate steps in your response."},
                        {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {self.question}"}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": self.agent.instructions + "\nIMPORTANT: Provide ONLY your final answer. Do NOT include your thinking process, reasoning, or intermediate steps in your response."},
                        {"role": "user", "content": self.question}
                    ]
                
                logger.info("Getting response from model using complete_chat")
                response = self.agent.model.complete_chat(messages)
                logger.info(f"Got response from model: {len(response)} characters")
            else:
                # Fall back to the standard agent method but capture the result
                # Temporarily redirect stdout to capture the output
                import io
                from contextlib import redirect_stdout
                
                f = io.StringIO()
                with redirect_stdout(f):
                    self.agent.print_response(self.question, markdown=True)
                
                output = f.getvalue()
                
                # Extract just the actual response text from the captured output
                # Split the output to remove headers/footers added by agno framework
                lines = output.split("\n")
                
                # Skip past headers to get to the response
                response_lines = []
                collecting = False
                found_response = False
                thinking_mode = False
                
                for line in lines:
                    # Skip sections that indicate thinking process
                    if any(marker in line.lower() for marker in ["thinking", "thought process", "reasoning", "let me think", "analyzing"]):
                        thinking_mode = True
                        continue
                        
                    # Exit thinking mode when we find a conclusion/result marker
                    if thinking_mode and any(marker in line.lower() for marker in ["answer:", "response:", "conclusion:", "therefore,"]):
                        thinking_mode = False
                    
                    # Start collecting after we see the Response header
                    if "Response" in line and "──────" in line:
                        collecting = True
                        found_response = True
                        continue
                        
                    # Stop collecting when we hit the end marker
                    if collecting and "───────────────────" in line:
                        collecting = False
                        continue
                        
                    # Collect the actual response text, skipping thinking sections
                    if collecting and not thinking_mode:
                        # Remove any leading/trailing whitespace and artifacts
                        clean_line = line.strip()
                        # Strip ANSI color codes, border characters, and other artifacts
                        clean_line = self._clean_ansi_and_formatting(clean_line)
                        if clean_line:  # Skip empty lines
                            response_lines.append(clean_line)
                
                # If we couldn't parse the output, just use it as is
                if not found_response or not response_lines:
                    logger.warning("Could not extract clean response text, using raw output")
                    response = self._clean_ansi_and_formatting(output)
                    
                    # Final cleanup to remove thinking process markers and sections
                    response_parts = response.split("\n")
                    final_parts = []
                    skip_section = False
                    
                    for part in response_parts:
                        if any(marker in part.lower() for marker in ["thinking", "thought process", "reasoning", "let me think", "analyzing"]):
                            skip_section = True
                            continue
                            
                        if skip_section and any(marker in part.lower() for marker in ["answer:", "response:", "conclusion:", "therefore,"]):
                            skip_section = False
                            # Keep this line as it likely has the answer
                            part = part.replace("Answer:", "").replace("Response:", "").replace("Conclusion:", "").strip()
                            final_parts.append(part)
                            continue
                            
                        if not skip_section:
                            final_parts.append(part)
                    
                    response = "\n".join(final_parts)
                else:
                    # Join the cleaned response lines
                    response = "\n".join(response_lines)
                
                logger.info(f"Processed response text (length: {len(response)} characters)")
            
            if not response or len(response.strip()) < 5:
                raise ValueError("Resposta vazia ou muito curta recebida do modelo")
            
            # Final cleanup to ensure no "thinking" sections remain
            response_parts = response.split("\n")
            final_response = []
            in_thinking_section = False
            
            for part in response_parts:
                # Check for thinking section headers
                if any(marker in part.lower() for marker in ["thinking:", "thought process:", "reasoning:", "let me think", "analyzing"]):
                    in_thinking_section = True
                    continue
                
                # Check for end of thinking section
                if in_thinking_section and any(marker in part.lower() for marker in ["answer:", "response:", "conclusion:", "therefore,"]):
                    in_thinking_section = False
                    # Extract just the answer part
                    for prefix in ["answer:", "response:", "conclusion:", "therefore,"]:
                        if prefix in part.lower():
                            part = part[part.lower().find(prefix) + len(prefix):].strip()
                            break
                
                # Add lines that aren't in thinking sections
                if not in_thinking_section:
                    final_response.append(part)
            
            response = "\n".join(final_response)
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
    status_update = pyqtSignal(str)  # New signal for status updates

    def __init__(self, vector_db):
        super().__init__()
        self.vector_db = vector_db
        self.is_running = True
        self.memory_threshold = 1000  # MB
        self.docs_dir = Path('docs')
        self.file_extensions = {
            'pdf': self.extract_text_from_pdf,
            'txt': self.extract_text_from_txt,
            'docx': None,  # Not implemented yet
            'doc': None     # Not implemented yet
        }

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                total_pages = len(reader.pages)
                
                # Show progress even inside the PDF processing
                for i, page in enumerate(reader.pages):
                    if not self.is_running:
                        break
                    text += page.extract_text() + "\n"
                    
                    # Update status for larger PDFs
                    if total_pages > 10 and i % 5 == 0:
                        self.status_update.emit(f"Processando {pdf_path.name}: página {i+1}/{total_pages}")
                        
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            raise

    def extract_text_from_txt(self, txt_path):
        """Extract text from text file"""
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='replace') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from {txt_path}: {str(e)}")
            raise

    def process_text(self, text, chunk_size=1000, overlap=100):
        """
        Split text into chunks with overlap to maintain context
        
        Args:
            text: Text to process
            chunk_size: Target size for each chunk
            overlap: Number of words to overlap between chunks
        """
        words = text.split()
        total_words = len(words)
        chunks = []
        
        if total_words == 0:
            return chunks
            
        # For very short texts, just return as is
        if total_words < chunk_size:
            chunks.append(" ".join(words))
            return chunks
            
        # Process into overlapping chunks
        pos = 0
        while pos < total_words:
            end_pos = min(pos + chunk_size, total_words)
            chunk = " ".join(words[pos:end_pos])
            chunks.append(chunk)
            
            # Move position forward, accounting for overlap
            pos += chunk_size - overlap
            
            # Avoid getting stuck at the end with too small chunks
            if end_pos == total_words and len(chunks) > 1:
                break
                
            # Show progress for very large documents
            if total_words > 10000 and len(chunks) % 10 == 0:
                progress = int(end_pos / total_words * 100)
                self.status_update.emit(f"Dividindo texto em partes: {progress}%")

        return chunks

    def run(self):
        try:
            # Force garbage collection before loading
            gc.collect()
            
            # Ensure docs directory exists
            self.docs_dir.mkdir(exist_ok=True)
            
            # Get list of supported files
            all_files = []
            for ext in self.file_extensions:
                if self.file_extensions[ext]:  # Only include implemented handlers
                    all_files.extend(list(self.docs_dir.glob(f'*.{ext}')))
            
            if not all_files:
                self.error.emit("Nenhum arquivo suportado encontrado no diretório 'docs'")
                return
                
            self.status_update.emit(f"Encontrados {len(all_files)} arquivos")

            # Process each file individually
            for i, file_path in enumerate(all_files):
                if not self.is_running:
                    break
                    
                try:
                    # Check memory usage
                    memory_usage = get_memory_usage()
                    if memory_usage > self.memory_threshold:
                        warning_msg = f"Alto uso de memória: {memory_usage:.1f}MB"
                        self.memory_warning.emit(warning_msg)
                        gc.collect()
                        time.sleep(1)

                    file_extension = file_path.suffix.lower()[1:]  # Remove the dot
                    if file_extension not in self.file_extensions or not self.file_extensions[file_extension]:
                        self.status_update.emit(f"Formato não suportado: {file_path.name}")
                        continue
                        
                    logger.info(f"Processing file: {file_path.name}")
                    self.file_processed.emit(f"Processando: {file_path.name}")
                    self.status_update.emit(f"Extraindo texto de {file_path.name}")

                    # Extract text using the appropriate handler
                    text = self.file_extensions[file_extension](file_path)
                    
                    # Process text into chunks
                    self.status_update.emit(f"Dividindo {file_path.name} em partes")
                    chunks = self.process_text(text)
                    
                    if not chunks:
                        self.status_update.emit(f"Nenhum texto extraído de {file_path.name}")
                        continue
                        
                    self.status_update.emit(f"Indexando {len(chunks)} partes de {file_path.name}")
                    
                    # Store chunks in vector database
                    for j, chunk in enumerate(chunks):
                        # Generate a unique ID for each chunk
                        chunk_id = str(uuid.uuid4())
                        
                        # Create metadata
                        metadata = {
                            "source": file_path.name,
                            "chunk_id": chunk_id,
                            "chunk_index": j,
                            "total_chunks": len(chunks)
                        }
                        
                        # Add to vector database
                        self.vector_db.add(
                            texts=[chunk],
                            metadatas=[metadata],
                            ids=[chunk_id]
                        )
                        
                        # Update status for large documents
                        if len(chunks) > 20 and j % 10 == 0:
                            progress_chunk = int((j + 1) / len(chunks) * 100)
                            self.status_update.emit(f"Indexando {file_path.name}: {progress_chunk}%")
                    
                    # Update overall progress
                    progress = int((i + 1) / len(all_files) * 100)
                    self.progress.emit(progress)
                    
                    # Force garbage collection after each file
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path.name}: {str(e)}")
                    self.error.emit(f"Erro ao processar {file_path.name}: {str(e)}")
                    continue

            if self.is_running:
                self.finished.emit()
                
        except Exception as e:
            logger.error(f"Error in document loading: {str(e)}")
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False

class SettingsView(QWidget):
    """Settings view for the application"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
        
    def initUI(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Header with title and back button
        header_layout = QHBoxLayout()
        
        # Back button
        self.back_button = QPushButton("Voltar")
        self.back_button.setIcon(QIcon.fromTheme("go-previous"))
        self.back_button.clicked.connect(self.go_back)
        self.back_button.setFont(QFont("Segoe UI", 11))
        self.back_button.setMinimumHeight(40)
        # Remove fixed width to show text properly
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f2f5;
                color: #515769;
                border: 1px solid #e0e4e8;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e9ecf2;
            }
            QPushButton:pressed {
                background-color: #e0e4e8;
            }
        """)
        header_layout.addWidget(self.back_button, alignment=Qt.AlignmentFlag.AlignLeft)
        
        # Title
        settings_title = QLabel("Configurações")
        settings_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        settings_title.setStyleSheet("color: #4a6fc3; margin-bottom: 5px;")
        header_layout.addWidget(settings_title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Add empty widget for spacing
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Separator line
        separator = QLabel()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #e0e4e8; margin-top: 5px; margin-bottom: 20px;")
        layout.addWidget(separator)
        
        # Settings content with placeholders for future development
        settings_content = QVBoxLayout()
        
        # Section: API Configuration
        api_section = QVBoxLayout()
        api_title = QLabel("Configuração de API")
        api_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        api_title.setStyleSheet("color: #515769; margin-bottom: 10px;")
        api_section.addWidget(api_title)
        
        api_info = QLabel("Estas configurações serão implementadas em versões futuras.")
        api_info.setStyleSheet("color: #939aab; margin-bottom: 20px;")
        api_section.addWidget(api_info)
        
        settings_content.addLayout(api_section)
        
        # Section: Document Processing
        doc_section = QVBoxLayout()
        doc_title = QLabel("Processamento de Documentos")
        doc_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        doc_title.setStyleSheet("color: #515769; margin-bottom: 10px;")
        doc_section.addWidget(doc_title)
        
        doc_info = QLabel("Opções para ajustar como os documentos são processados e armazenados.")
        doc_info.setStyleSheet("color: #939aab; margin-bottom: 20px;")
        doc_section.addWidget(doc_info)
        
        settings_content.addLayout(doc_section)
        
        # Section: Application Settings
        app_section = QVBoxLayout()
        app_title = QLabel("Configurações do Aplicativo")
        app_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        app_title.setStyleSheet("color: #515769; margin-bottom: 10px;")
        app_section.addWidget(app_title)
        
        app_info = QLabel("Opções para personalizar a aparência e comportamento do aplicativo.")
        app_info.setStyleSheet("color: #939aab; margin-bottom: 20px;")
        app_section.addWidget(app_info)
        
        app_version = QLabel(f"Versão do Aplicativo: {APP_VERSION}")
        app_version.setStyleSheet("color: #515769; font-weight: bold;")
        app_section.addWidget(app_version)
        
        settings_content.addLayout(app_section)
        
        layout.addLayout(settings_content)
        
        # Add stretching space at the bottom
        layout.addStretch()
        
    def go_back(self):
        """Return to the main view"""
        if self.parent and hasattr(self.parent, 'stacked_widget'):
            # Switch to main view (index 0)
            self.parent.stacked_widget.setCurrentIndex(0)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} v{APP_VERSION}")
        self.resize(900, 700)
        
        # Initialize persistence
        self.persistence = VectorStorePersistence(APP_DATA_DIR)
        
        # Initialize chat history
        self.chat_history = ChatHistory()
        
        # Show splash screen during initialization
        self._show_splash_screen()
        
        # Setup stacked widget to hold main view and settings view
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create main content widget
        self.main_widget = QWidget()
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create settings view
        self.settings_view = SettingsView(self)
        
        # Add both widgets to the stacked widget
        self.stacked_widget.addWidget(self.main_widget)
        self.stacked_widget.addWidget(self.settings_view)
        
        # Set main view as default
        self.stacked_widget.setCurrentIndex(0)
        
        try:
            # Initialize AI components
            self._init_ai_components()
            
            # Create UI components
            self._create_ui_components(main_layout)
            
            # Create status bar
            self._create_status_bar()
            
            # Initialize chat count
            self._update_chat_count()
            
            # Apply styles
            self._apply_styles()
            
            # Load persisted data
            self._load_persisted_data()
            
            # Ensure chat history is displayed
            QTimer.singleShot(200, self.update_history_view)
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            
            # Hide splash screen if error
            if hasattr(self, 'splash'):
                self.splash.finish(self)
            QMessageBox.critical(self, "Erro de Inicialização", 
                              f"Erro ao inicializar componentes: {str(e)}\n"
                              "Verifique se o arquivo .env está configurado corretamente.")

    def _show_splash_screen(self):
        """Show a splash screen during app initialization"""
        splash_pixmap = QPixmap(220, 120)  # Create a blank pixmap
        splash_pixmap.fill(Qt.GlobalColor.white)
        self.splash = QSplashScreen(splash_pixmap, Qt.WindowType.WindowStaysOnTopHint)
        
        # Add text to splash screen
        self.splash.showMessage(f"{APP_TITLE} v{APP_VERSION}\nIniciando...", 
                               Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.black)
        self.splash.show()
        QApplication.processEvents()

    def _init_ai_components(self):
        """Initialize AI components"""
        try:
            _ = load_dotenv(find_dotenv())
            
            # Initialize the embedder
            self.embedder = OpenAIEmbedder()
            
            # Initialize the vector store
            self.vector_db = SimpleVectorStore(self.embedder)

            self.agent = Agent(
                model=OpenAIChat(id='gpt-4o-mini'),
                knowledge=self.vector_db,
                search_knowledge=True,
                show_tool_calls=False,
                
                markdown=False,
                add_history_to_messages=True,
                instructions='''
                    "Você é um assistente especializado em analisar atas de reuniões em formato .pdf. Sempre que o usuário fizer uma pergunta,
                    primeiro reformule-a para torná-la mais clara e específica, com foco em palavras-chave, datas, pessoas ou eventos relevantes,
                    para otimizar a busca nos documentos. Após a reformulação, busque nas atas e forneça uma resposta precisa e objetiva.
                    
                    IMPORTANTE: NÃO inclua seu processo de pensamento, reformulações, análises ou explicações sobre como chegou à resposta.
                    Simplesmente forneça a resposta final diretamente, sem mencionar como fez a busca ou como processou a informação.
                    
                    Faça buscas adicionais para garantir que todas as partes da questão sejam abordadas. Responda apenas o que tiver relacionado a pergunta, sem informações extras desnecessárias.
                    
                    Lembre-se: Apresente APENAS os resultados/resposta final, nunca seu raciocínio ou processo de pensamento."
                '''
            )
            
            # Hide splash screen after components are loaded
            if hasattr(self, 'splash'):
                self.splash.finish(self)
                
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            # Hide splash screen if error
            if hasattr(self, 'splash'):
                self.splash.finish(self)
            QMessageBox.critical(self, "Erro de Inicialização", 
                              f"Erro ao inicializar componentes: {str(e)}\n"
                              "Verifique se o arquivo .env está configurado corretamente.")

    def _create_ui_components(self, layout):
        """Create and arrange UI components"""
        # Define a common font for buttons
        button_font = QFont("Arial", 11)
        
        # History layout - Now first section
        self.history_layout = QVBoxLayout()
        
        # Container for the scrollable history
        history_container = QVBoxLayout()
        
        # Header with controls for history
        history_header = QHBoxLayout()
        
        # History title with modern font
        history_title = QLabel("Histórico de Conversas")
        history_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        history_title.setStyleSheet("color: #4a6fc3; margin-bottom: 5px;")
        history_header.addWidget(history_title)
        
        # Add stretch to push buttons to the right
        history_header.addStretch()
        
        # Clear history button with red X emoji
        self.clear_history_button = QPushButton("❌  Limpar Chat")  # Red X emoji
        self.clear_history_button.setToolTip("Limpar Histórico")
        self.clear_history_button.clicked.connect(self.clear_chat_history)
        self.clear_history_button.setFixedSize(160, 32)  # Increased width from 140 to 160
        self.clear_history_button.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        
        self.clear_history_button.setStyleSheet("""
            QPushButton {
                background-color: #ffeded;
                color: #e53e3e;
                border: 1px solid #e53e3e;
                border-radius: 4px;
                padding: 4px 12px;
                margin: 2px;
                text-align: center;
                qproperty-alignment: AlignCenter;
                font-family: 'Segoe UI', 'Roboto', 'Open Sans', sans-serif;
            }
            QPushButton:hover {
                background-color: #ffe0e0;
                border: 1px solid #c53030;
                color: #c53030;
            }
            QPushButton:pressed {
                background-color: #ffd0d0;
                border: 1px solid #9b2c2c;
                color: #9b2c2c;
            }
        """)
        history_header.addWidget(self.clear_history_button)
        
        # Settings button with icon and modern font
        self.settings_button = QPushButton()
        self.settings_button.setToolTip("Configurações")
        self.settings_button.clicked.connect(self.show_settings)
        # Set fixed size for a smaller button
        self.settings_button.setFixedSize(30, 30)
        # Direct use of icon character in button text
        self.settings_button.setText("⚙")
        # Set smaller font size for better proportions
        self.settings_button.setFont(QFont("Segoe UI", 13))
        # Make icon centered in button
        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #5c85d6;
                border: none;
                padding: 0;
                margin: 2px;
                text-align: center;
                font-family: 'Segoe UI', 'Roboto', 'Open Sans', sans-serif;
            }
            QPushButton:hover {
                color: #4a6fc3;
                background-color: rgba(92, 133, 214, 0.1);
            }
            QPushButton:pressed {
                color: #3a5fb3;
            }
        """)
        history_header.addWidget(self.settings_button)
        
        history_container.addLayout(history_header)
        
        # Scroll area for history content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Container widget for history content
        self.history_content = QTextEdit()
        self.history_content.setReadOnly(True)
        self.history_content.setAcceptRichText(True)
        
        # Configure history text rendering options
        history_document = self.history_content.document()
        history_document.setDefaultStyleSheet("""
            body {
                font-family: 'Segoe UI', 'Roboto', 'Open Sans', sans-serif;
                margin: 0;
                padding: 0;
                font-size: 12px;
                background-color: #f8f9fa;
            }
            .date-header {
                background-color: #eef2f9;
                color: #5c85d6;
                font-weight: bold;
                padding: 10px 14px;
                margin: 20px 0 12px 0;
                border-radius: 6px;
                border-left: 4px solid #5c85d6;
                font-size: 18px;
            }
            .chat-entry {
                margin-bottom: 20px;
                padding: 16px;
                border: 1px solid #e0e4e8;
                border-radius: 8px;
                background-color: white;
            }
            .timestamp {
                color: #939aab;
                font-size: 12pt;
                margin-bottom: 8px;
                text-align: right;
            }
            .question, .answer {
                margin-bottom: 10px;
            }
            .q-label, .a-label {
                font-weight: bold;
                margin-bottom: 4px;
                font-size: 17px;
            }
            .q-label {
                color: #38a169;
            }
            .a-label {
                color: #dd6b20;
            }
            .q-content, .a-content {
                margin-left: 10px;
                line-height: 1.6;
                padding: 6px 0;
                font-size: 16px;
            }
            .q-content {
                background-color: rgba(56, 161, 105, 0.05);
                border-left: 3px solid #38a169;
                padding-left: 12px;
                border-radius: 0 4px 4px 0;
            }
            .a-content {
                background-color: rgba(221, 107, 32, 0.05);
                border-left: 3px solid #dd6b20;
                padding-left: 12px;
                border-radius: 0 4px 4px 0;
            }
        """)
        
        # Set initial empty state
        self.history_content.setHtml("""
        <!DOCTYPE html>
        <html>
        <head></head>
        <body>
            <div style="text-align: center; margin-top: 40px; color: #939aab; font-size: 14px;">
                <div style="margin-bottom: 10px;">✨ Histórico de conversas vazio ✨</div>
                <div>Comece uma nova conversa fazendo uma pergunta!</div>
            </div>
        </body>
        </html>
        """)
        
        # Add to scroll area
        scroll_area.setWidget(self.history_content)
        history_container.addWidget(scroll_area)
        
        self.history_layout.addLayout(history_container)
        
        # Add the history panel to the main layout
        layout.addLayout(self.history_layout)
        
        # Separator line
        separator = QLabel()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #e0e4e8; margin-top: 5px; margin-bottom: 10px;")
        layout.addWidget(separator)
        
        # Question input with modern font
        question_label = QLabel("Faça sua pergunta sobre as atas:")
        question_label.setFont(QFont("Segoe UI", 14))
        question_label.setStyleSheet("color: #4a6fc3; margin-top: 10px; font-weight: bold;")
        layout.addWidget(question_label)
        
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("Digite sua pergunta aqui...")
        self.question_input.setMaximumHeight(100)
        self.question_input.setFont(QFont("Segoe UI", 14))
        self.question_input.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0e4e8;
                border-radius: 8px;
                padding: 10px;
                background-color: white;
                color: #363a43;
                font-family: 'Segoe UI', 'Roboto', 'Open Sans', sans-serif;
            }
            QTextEdit:focus {
                border: 1px solid #5c85d6;
                background-color: #fafbfc;
            }
        """)
        layout.addWidget(self.question_input)
        
        # Buttons with modern font
        button_layout = QHBoxLayout()
        
        self.ask_button = QPushButton("Perguntar")
        self.ask_button.setIcon(QIcon.fromTheme("dialog-question"))
        self.ask_button.clicked.connect(self.process_question)
        self.ask_button.setEnabled(False)  # Disabled until documents are loaded
        self.ask_button.setFont(QFont("Segoe UI", 11))
        self.ask_button.setMinimumHeight(40)
        button_layout.addWidget(self.ask_button)
        
        self.load_docs_button = QPushButton("Carregar Documentos")
        self.load_docs_button.setIcon(QIcon.fromTheme("document-open"))
        self.load_docs_button.clicked.connect(self.load_documents)
        self.load_docs_button.setFont(QFont("Segoe UI", 11))
        self.load_docs_button.setMinimumHeight(40)
        button_layout.addWidget(self.load_docs_button)
        
        layout.addLayout(button_layout)
        
        # Hidden response output for storing responses
        # This is needed for the backend but not shown in the UI
        self.response_output = QTextEdit()
        self.response_output.setReadOnly(True)
        self.response_output.setAcceptRichText(True)
        self.response_output.setVisible(False)

    def _create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create document status label (but not displayed in header)
        self.doc_status_label = QLabel("Sem documentos carregados")
        self.doc_status_label.setStyleSheet("color: #939aab;")
        
        # Add document counter to status bar
        self.doc_count_label = QLabel("Documentos: 0")
        self.status_bar.addPermanentWidget(self.doc_count_label)
        
        # Add chat history counter to status bar
        self.chat_count_label = QLabel("Conversas: 0")
        self.status_bar.addPermanentWidget(self.chat_count_label)
        
        # Add memory usage to status bar
        self.memory_label = QLabel("Memória: 0 MB")
        self.status_bar.addPermanentWidget(self.memory_label)
        
        # Start memory usage timer
        self.memory_timer = QTimer(self)
        self.memory_timer.timeout.connect(self._update_memory_usage)
        self.memory_timer.start(5000)  # Update every 5 seconds
        
    def _update_memory_usage(self):
        """Update memory usage display"""
        try:
            memory_mb = get_memory_usage()
            self.memory_label.setText(f"Memória: {memory_mb:.1f} MB")
            
            # Change color based on memory usage
            if memory_mb > 500:
                self.memory_label.setStyleSheet("color: #e53e3e; font-weight: bold;")  # High memory use (red)
            elif memory_mb > 300:
                self.memory_label.setStyleSheet("color: #dd6b20; font-weight: bold;")  # Medium memory use (orange)
            else:
                self.memory_label.setStyleSheet("color: #515769;")                     # Normal memory use
        except Exception as e:
            logger.error(f"Error updating memory usage: {e}")
            
    def _apply_styles(self):
        """Apply styles to components"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QTextEdit {
                border: 1px solid #e0e4e8;
                border-radius: 8px;
                padding: 10px;
                background-color: white;
                color: #363a43;
                font-family: 'Segoe UI', 'Roboto', 'Open Sans', sans-serif;
            }
            QPushButton {
                background-color: #5c85d6;
                color: white;
                border: none;
                padding: 10px 18px;
                border-radius: 6px;
                font-weight: bold;
                font-family: 'Segoe UI', 'Roboto', 'Open Sans', sans-serif;
            }
            QPushButton:hover {
                background-color: #4a6fc3;
            }
            QPushButton:pressed {
                background-color: #3a5fb3;
            }
            QPushButton:disabled {
                background-color: #d8dde5;
                color: #939aab;
            }
            QLabel {
                font-weight: bold;
                color: #363a43;
                font-family: 'Segoe UI', 'Roboto', 'Open Sans', sans-serif;
            }
            QProgressDialog {
                background-color: white;
                border-radius: 6px;
                font-family: 'Segoe UI', 'Roboto', 'Open Sans', sans-serif;
            }
            QStatusBar {
                background-color: #e9ecf2;
                color: #515769;
                border-top: 1px solid #d0d4db;
                font-family: 'Segoe UI', 'Roboto', 'Open Sans', sans-serif;
            }
        """)

    def _load_persisted_data(self):
        """Load persisted data from disk"""
        try:
            # Show loading status
            self.status_bar.showMessage("Carregando dados persistidos...")
            QApplication.processEvents()
            
            # Load vector db from persistence
            success = self.persistence.load_vector_store(self.vector_db)
            
            # Load chat history
            chat_history_file = os.path.join(APP_DATA_DIR, "chat_history.json")
            self.chat_history.load_from_disk(chat_history_file)
            
            # Update chat count after loading history
            self._update_chat_count()
            
            # Explicitly update history view to ensure it shows
            self.update_history_view()
            
            # Update UI based on results
            if success:
                # Count unique file sources in metadata instead of total chunks
                sources = set()
                for metadata in self.vector_db.metadatas:
                    if "source" in metadata:
                        sources.add(metadata["source"])
                
                file_count = len(sources)
                chunk_count = len(self.vector_db.documents)
                
                # Document count available in persistence.document_count if loaded
                if file_count > 0:
                    self._update_document_status(f"{file_count} documentos carregados")
                    self.doc_count_label.setText(f"Documentos: {file_count}")
                    self.ask_button.setEnabled(True)
                else:
                    self._update_document_status("Nenhum documento encontrado na base de dados")
                    self.ask_button.setEnabled(False)
                    
                self.status_bar.showMessage("Dados carregados com sucesso!", 3000)
            else:
                self._update_document_status("Nenhum documento encontrado na base de dados")
                self.status_bar.showMessage("Nenhum dado persistido encontrado", 3000)
                self.ask_button.setEnabled(False)
                
        except Exception as e:
            logger.error(f"Error loading persisted data: {str(e)}")
            self.status_bar.showMessage("Erro ao carregar dados persistidos", 3000)
            self._update_document_status("Erro ao carregar documentos")

    def _update_document_status(self, text):
        """Update document status display"""
        # Update only the status bar message since the label is not displayed anymore
        self.status_bar.showMessage(text, 3000)

    def process_question(self):
        """Process the user's question"""
        question = self.question_input.toPlainText().strip()
        if not question:
            QMessageBox.warning(self, "Aviso", "Por favor, digite uma pergunta.")
            return
        
        # Temporarily add the question to history with a placeholder response
        timestamp = QDateTime.currentDateTime()
        temp_entry = {
            "timestamp": timestamp,
            "question": question,
            "answer": "⏳ Processando sua pergunta..."
        }
        
        # Add temporary entry to the history
        self.chat_history.history.append(temp_entry)
        
        # Update the history view with the new question immediately
        self.update_history_view()
        
        # Scroll to the bottom of the history view to show the new question
        scrollbar = self.history_content.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Disable ask button during processing
        self.ask_button.setEnabled(False)
        self.status_bar.showMessage("Processando pergunta...")
        
        # Clear and set placeholder in the response area
        self.response_output.clear()
        self.response_output.setTextColor(QColor("#939aab"))
        self.response_output.append("⏳ Processando sua pergunta...")
        
        # Create and start worker thread
        self.worker = WorkerThread(self.agent, question)
        self.worker.response_ready.connect(self.handle_response)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(lambda: self._on_question_finished())
        self.worker.start()
        
        # Clear the question input text
        self.question_input.setPlainText("")

    def _on_question_finished(self):
        """Handle question processing completion"""
        self.ask_button.setEnabled(True)
        self.status_bar.showMessage("Resposta pronta", 3000)

    def handle_response(self, response):
        """Handle the AI response"""
        if not response:
            self.handle_error("Nenhuma resposta recebida do servidor")
            return
            
        try:
            # Log response for debugging
            logger.info(f"Displaying response (length: {len(response)})")
            
            # Find and update the temporary entry with the actual response
            if self.chat_history.history:
                # Get the last entry in history (which should be our temporary entry)
                last_entry = self.chat_history.history[-1]
                if "⏳ Processando sua pergunta..." in last_entry["answer"]:
                    # Replace the placeholder with the actual response
                    self.chat_history.history[-1]["answer"] = response
                else:
                    # If not found with placeholder, use the question from last entry
                    question = last_entry["question"]
                    self.chat_history.add_entry(question, response)
            
            # Update the history view in real-time
            self.update_history_view()
            
            # Clear previous content in response area
            self.response_output.clear()
            
            # Convert the text to HTML with enhanced formatting
            html_content = self._format_response_to_html(response)
            
            # Set the HTML content directly
            self.response_output.setHtml(html_content)
            
            # Move cursor to start
            cursor = self.response_output.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            self.response_output.setTextCursor(cursor)
            
            # Update UI
            QApplication.processEvents()
            
        except Exception as e:
            logger.error(f"Error displaying response: {str(e)}")
            self.handle_error(f"Erro ao exibir resposta: {str(e)}")
            
    def _format_response_to_html(self, text):
        """Convert the response text to rich HTML with enhanced formatting"""
        # First clean any ANSI and formatting artifacts
        clean_text = self._clean_ansi_and_formatting(text.strip())
        
        # Convert markdown to HTML with simpler, more robust formatting
        html = self._simple_text_to_html(clean_text)
        
        # Wrap in a simple HTML document with clean styling
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 12pt;
                    line-height: 1.5;
                    color: #333;
                    margin: 0;
                    padding: 10px;
                }}
                p {{
                    margin: 0 0 12px 0;
                }}
                h2, h3, h4 {{
                    color: #2a6fc9;
                    margin-top: 16px;
                    margin-bottom: 8px;
                }}
                ul, ol {{
                    margin-bottom: 12px;
                }}
                li {{
                    margin-bottom: 4px;
                }}
                .content {{
                    max-width: 100%;
                }}
            </style>
        </head>
        <body>
            <div class="content">
                {html}
            </div>
        </body>
        </html>
        """
    
    def _simple_text_to_html(self, text):
        """Convert text to HTML with simple, reliable formatting"""
        # Replace special characters
        html = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Split into paragraphs
        paragraphs = html.split('\n\n')
        formatted_paragraphs = []
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            # Check if this is a bullet list (starts with *, -, or •)
            if paragraph.strip().startswith(('* ', '- ', '• ')):
                # Process as a bullet list
                items = paragraph.split('\n')
                list_html = '<ul>\n'
                for item in items:
                    item = item.strip()
                    if item.startswith(('* ', '- ', '• ')):
                        # Extract the text after the bullet
                        item_text = item[2:].strip()
                        list_html += f'<li>{item_text}</li>\n'
                list_html += '</ul>'
                formatted_paragraphs.append(list_html)
                
            # Check if this is a numbered list (starts with 1., 2., etc.)
            elif paragraph.strip() and paragraph.strip()[0].isdigit() and paragraph.strip().find('. ') > 0:
                # Process as a numbered list
                items = paragraph.split('\n')
                list_html = '<ol>\n'
                for item in items:
                    item = item.strip()
                    if item and item[0].isdigit() and item.find('. ') > 0:
                        # Extract the text after the number
                        item_text = item[item.find('. ')+2:].strip()
                        list_html += f'<li>{item_text}</li>\n'
                list_html += '</ol>'
                formatted_paragraphs.append(list_html)
                
            # Check if this is a heading (starts with # or ##)
            elif paragraph.strip().startswith('#'):
                line = paragraph.strip()
                if line.startswith('# '):
                    heading_text = line[2:]
                    formatted_paragraphs.append(f'<h2>{heading_text}</h2>')
                elif line.startswith('## '):
                    heading_text = line[3:]
                    formatted_paragraphs.append(f'<h3>{heading_text}</h3>')
                elif line.startswith('### '):
                    heading_text = line[4:]
                    formatted_paragraphs.append(f'<h4>{heading_text}</h4>')
                else:
                    # Treat as regular paragraph
                    formatted_paragraphs.append(f'<p>{paragraph.replace("\n", "<br>")}</p>')
            else:
                # Regular paragraph - replace newlines with <br>
                formatted_paragraphs.append(f'<p>{paragraph.replace("\n", "<br>")}</p>')
        
        # Process text formatting within the HTML
        result = '\n'.join(formatted_paragraphs)
        
        # Basic formatting: bold and italic
        # Bold: replace **text** or __text__ with <strong>text</strong>
        result = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', result)
        result = re.sub(r'__(.+?)__', r'<strong>\1</strong>', result)
        
        # Italic: replace *text* or _text_ with <em>text</em>
        result = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', result)
        result = re.sub(r'_([^_]+)_', r'<em>\1</em>', result)
        
        return result

    def _clean_ansi_and_formatting(self, text):
        """Remove ANSI color codes, box drawing characters and other formatting artifacts"""
        import re
        
        if not text:
            return ""
            
        # Remove ANSI escape sequences (color codes)
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        result = ansi_escape.sub('', text)
        
        # Remove box drawing and formatting characters (pipes, etc.)
        result = re.sub(r'[│┌┐└┘┐┘─┬┴┼┤├]', '', result)
        
        # Remove any remaining formatting artifacts
        result = re.sub(r'\[3[0-9]m', '', result)  # Color markers like [34m
        result = re.sub(r'\[0m', '', result)       # Reset markers
        
        # Further cleanup of any artifacts
        return result.strip()

    def handle_error(self, error_message):
        """Handle an error message"""
        logging.error(f"Error occurred: {error_message}")
        
        # Update response area with error
        self.response_output.clear()
        self.response_output.setTextColor(QColor("#e53e3e"))
        self.response_output.append(f"Erro: {error_message}")
        
        # Find and update the temporary entry with the error message
        if self.chat_history.history:
            # Get the last entry in history (which should be our temporary entry)
            last_entry = self.chat_history.history[-1]
            if "⏳ Processando sua pergunta..." in last_entry["answer"]:
                # Replace the placeholder with the error message
                error_html = f'<span style="color: #e53e3e;">Erro: {error_message}</span>'
                self.chat_history.history[-1]["answer"] = error_html
                
                # Update the history view
                self.update_history_view()
        
        # Reset UI state
        self.ask_button.setEnabled(True)
        self.status_bar.showMessage(f"Erro: {error_message}", 5000)

    def load_documents(self):
        """Load PDF documents"""
        try:
            # Disable buttons during loading
            self.load_docs_button.setEnabled(False)
            self.ask_button.setEnabled(False)
            
            # Create progress dialog
            self.progress = QProgressDialog("Carregando documentos...", "Cancelar", 0, 100, self)
            self.progress.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress.setCancelButton(None)  # Better UX to not allow cancellation
            self.progress.setMinimumDuration(0)
            self.progress.setAutoClose(False)
            self.progress.setAutoReset(False)
            self.progress.show()
            
            # Update status
            self._update_document_status("Carregando documentos...")
            self.status_bar.showMessage("Carregando e indexando documentos...")
            
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
        self.status_bar.showMessage(message)
        self.memory_label.setText(message)
        self.memory_label.setStyleSheet("color: red; font-weight: bold;")

    def _on_documents_loaded(self):
        """Handle successful document loading"""
        self.progress.close()
        self.load_docs_button.setEnabled(True)
        self.ask_button.setEnabled(True)
        
        # Count unique file sources in metadata instead of total chunks
        sources = set()
        for metadata in self.vector_db.metadatas:
            if "source" in metadata:
                sources.add(metadata["source"])
        
        file_count = len(sources)
        chunk_count = len(self.vector_db.documents)
        
        # Update status - show file count, not chunk count
        self._update_document_status(f"{file_count} documentos carregados")
        self.doc_count_label.setText(f"Documentos: {file_count}")
        
        # Show success message
        QMessageBox.information(self, "Sucesso", f"Documentos carregados com sucesso! ({file_count} arquivos, {chunk_count} fragmentos)")
        self.status_bar.showMessage("Documentos carregados com sucesso!", 5000)
        
        # Save the vector store to disk
        self.persistence.save_vector_store(self.vector_db)

    def _on_document_load_error(self, error_message, progress=None):
        """Handle document loading errors"""
        if progress:
            progress.close()
        self.load_docs_button.setEnabled(True)
        self.ask_button.setEnabled(True)
        self._update_document_status("Erro ao carregar documentos")
        QMessageBox.critical(self, "Erro", f"Erro ao carregar documentos: {error_message}")
        self.status_bar.showMessage(f"Erro: {error_message}", 5000)
        
        
    def closeEvent(self, event):
        """Handle the window close event"""
        # Save persistence data
        try:
            self.status_bar.showMessage("Salvando dados...")
            
            # Save vector store
            if hasattr(self, 'vector_db') and self.vector_db and len(self.vector_db.documents) > 0:
                self.persistence.save_vector_store(self.vector_db)
                
            # Save chat history
            if hasattr(self, 'chat_history') and self.chat_history and len(self.chat_history.history) > 0:
                chat_history_file = os.path.join(APP_DATA_DIR, "chat_history.json")
                self.chat_history.save_to_disk(chat_history_file)
                
        except Exception as e:
            logger.error(f"Error saving data on close: {str(e)}")
            QMessageBox.warning(self, "Erro ao salvar", f"Erro ao salvar dados: {str(e)}")
            
        # Accept the close event
        event.accept()

    def update_history_view(self):
        """Update the history view with current chat history"""
        # Save current scroll position
        scrollbar = self.history_content.verticalScrollBar()
        current_position = scrollbar.value()
        
        # Clear current content first
        self.history_content.clear()
        
        # Get formatted history
        formatted_history = self.chat_history.get_formatted_history()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head></head>
        <body>
            {formatted_history}
        </body>
        </html>
        """
        self.history_content.setHtml(html_content)
        
        # Update the chat count in the status bar
        self._update_chat_count()
        
        # Restore previous scroll position
        scrollbar.setValue(current_position)

    def clear_chat_history(self):
        """Clear the chat history"""
        confirm = QMessageBox.question(
            self,
            "Limpar Histórico",
            "Tem certeza que deseja limpar todo o histórico de conversas?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            # Clear the history object
            self.chat_history.clear_history()
            
            # Remove history file from disk if it exists
            chat_history_file = os.path.join(APP_DATA_DIR, "chat_history.json")
            try:
                if os.path.exists(chat_history_file):
                    os.remove(chat_history_file)
                    logger.info(f"Removed chat history file: {chat_history_file}")
            except Exception as e:
                logger.error(f"Error removing chat history file: {str(e)}")
            
            # Set empty state HTML to history content
            self.history_content.clear()
            self.history_content.setHtml("""
            <!DOCTYPE html>
            <html>
            <head></head>
            <body>
                <div style="text-align: center; margin-top: 40px; color: #939aab; font-size: 14px;">
                    <div style="margin-bottom: 10px;">✨ Histórico de conversas vazio ✨</div>
                    <div>Comece uma nova conversa fazendo uma pergunta!</div>
                </div>
            </body>
            </html>
            """)
            
            # Force garbage collection
            gc.collect()
            
            # Update the chat count
            self._update_chat_count()
            
            self.status_bar.showMessage("Histórico limpo com sucesso", 3000)
    
    def _update_chat_count(self):
        """Update the chat count label"""
        count = len(self.chat_history.history)
        self.chat_count_label.setText(f"Conversas: {count}")
        
        # Update the color based on count
        if count > 0:
            self.chat_count_label.setStyleSheet("color: #38a169;")
        else:
            self.chat_count_label.setStyleSheet("color: #515769;")

    def show_settings(self):
        """Show the settings view"""
        self.stacked_widget.setCurrentIndex(1)

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
        
    def __str__(self):
        """String representation for display purposes"""
        source = self.metadata.get('source', 'Desconhecido')
        return f"Documento: {source} (Score: {self.score:.2f if self.score else 'N/A'})"

class ChatHistory:
    """Class to store and manage chat history"""
    def __init__(self, max_entries=50):
        self.history = []
        self.max_entries = max_entries
        
    def add_entry(self, question, answer):
        """Add a question-answer pair to history with timestamp"""
        timestamp = QDateTime.currentDateTime()
        entry = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer
        }
        self.history.append(entry)
        
        # Remove oldest entries if exceeding max_entries
        if len(self.history) > self.max_entries:
            self.history = self.history[-self.max_entries:]
            
    def get_history(self):
        """Return the full history"""
        return self.history
        
    def clear_history(self):
        """Clear all history"""
        # Clear the history list completely
        self.history = []
        
        # Explicitly request garbage collection
        gc.collect()
        
    def get_formatted_history(self):
        """Return history formatted as HTML for display"""
        if not self.history:
            return """
            <div style="text-align: center; margin-top: 40px; color: #939aab; font-size: 14px;">
                <div style="margin-bottom: 10px;">✨ Histórico de conversas vazio ✨</div>
                <div>Comece uma nova conversa fazendo uma pergunta!</div>
            </div>
            """
            
        # Group entries by date
        entries_by_date = {}
        for entry in self.history:
            date_str = entry["timestamp"].toString("yyyy-MM-dd")
            if date_str not in entries_by_date:
                entries_by_date[date_str] = []
            entries_by_date[date_str].append(entry)
        
        html = ""
        # Process entries by date, oldest first
        for date_str in sorted(entries_by_date.keys()):
            # Format the date as a header
            display_date = QDateTime.fromString(date_str, "yyyy-MM-dd").toString("dd 'de' MMMM 'de' yyyy")
            html += f"""
            <div class="date-header">
                {display_date}
            </div>
            """
            
            # Add all conversations for this date, oldest first
            for entry in sorted(entries_by_date[date_str], key=lambda e: e["timestamp"]):
                time_str = entry["timestamp"].toString("hh:mm:ss")
                question = entry["question"].replace('\n', '<br>')
                answer = entry["answer"].replace('\n', '<br>')
                
                html += f"""
                <div class="chat-entry">
                    <div class="timestamp">{time_str}</div>
                    <div class="question">
                        <div class="q-label">Você:</div>
                        <div class="q-content">{question}</div>
                    </div>
                    <div class="answer">
                        <div class="a-label">Resposta:</div>
                        <div class="a-content">{answer}</div>
                    </div>
                </div>
                """
        return html
    
    def save_to_disk(self, file_path):
        """Save chat history to disk"""
        try:
            # Convert QDateTime objects to strings for JSON serialization
            serializable_history = []
            for entry in self.history:
                serializable_entry = {
                    "timestamp": entry["timestamp"].toString("yyyy-MM-ddThh:mm:ss"),
                    "question": entry["question"],
                    "answer": entry["answer"]
                }
                serializable_history.append(serializable_entry)
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Chat history saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            return False
            
    def load_from_disk(self, file_path):
        """Load chat history from disk"""
        try:
            if not os.path.exists(file_path):
                logger.info(f"No chat history file found at {file_path}")
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                serialized_history = json.load(f)
                
            # Convert string timestamps back to QDateTime objects
            self.history = []
            for entry in serialized_history:
                dt = QDateTime.fromString(entry["timestamp"], "yyyy-MM-ddThh:mm:ss")
                self.history.append({
                    "timestamp": dt,
                    "question": entry["question"],
                    "answer": entry["answer"]
                })
                
            logger.info(f"Loaded {len(self.history)} chat history entries from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading chat history: {str(e)}")
            return False

def main():
    """Main application entry point"""
    # Enable high DPI support
    try:
        # Try the newer PyQt6 style first
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        # Fall back to the older style for PyQt6 6.6.1+
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        # High DPI scaling is enabled by default in newer PyQt versions
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setApplicationVersion(APP_VERSION)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
