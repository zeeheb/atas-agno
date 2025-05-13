#!/usr/bin/env python
"""
ATAS Application Launcher with pre-loading
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main launcher function"""
    try:
        # Welcome message
        print("""
        ╔════════════════════════════════════════════╗
        ║        iATAS - Analisador de ATAS          ║
        ╚════════════════════════════════════════════╝
        
        Iniciando aplicação...
        """)
        
        # Check for API keys
        _ = load_dotenv(find_dotenv())
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n⚠️ AVISO: Chave de API do OpenAI não encontrada!")
            print("Por favor, configure a chave no arquivo .env")
            print("Os recursos de IA ficarão limitados.\n")
            
        # Check for docs directory
        docs_dir = Path("docs")
        if not docs_dir.exists():
            docs_dir.mkdir()
            print("\n📁 Diretório 'docs' criado")
            print("Coloque seus arquivos PDF neste diretório.\n")
        else:
            num_files = len(list(docs_dir.glob("*.pdf"))) + len(list(docs_dir.glob("*.txt")))
            if num_files > 0:
                print(f"\n📁 {num_files} arquivos encontrados no diretório 'docs'")
            else:
                print("\n⚠️ Nenhum arquivo encontrado no diretório 'docs'")
                print("Adicione arquivos PDF ou TXT para análise.\n")
            
        # Import main module (only after environment checks)
        from main import main as start_app
        
        # Start the application
        print("\n🚀 Iniciando a aplicação...\n")
        start_app()
        
    except Exception as e:
        logger.error(f"Error in launcher: {str(e)}", exc_info=True)
        print(f"\n❌ Erro ao iniciar aplicação: {str(e)}")
        print("Verifique o arquivo de log 'app.log' para mais detalhes.")
        input("\nPressione Enter para sair...")
        
if __name__ == "__main__":
    main() 