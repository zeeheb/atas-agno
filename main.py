from agno.agent import Agent
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv, find_dotenv
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.embedder.openai import OpenAIEmbedder
from agno.document.chunking.agentic import AgenticChunking


_ = load_dotenv(find_dotenv())

vector_db = ChromaDb(collection="pdf_documents",
                    path="tmp/chromadb", 
                    persistent_client=True, 
                    embedder=OpenAIEmbedder())

knowledge_base = PDFKnowledgeBase(
  path='docs',
  vector_db=vector_db,
  chunking_strategy=AgenticChunking(),
)

# knowledge_base.load(recreate=True)  # Comment out after first run

agent = Agent(
  model=OpenAIChat(id='gpt-4o-mini'),
  knowledge=knowledge_base,
  search_knowledge=True,
  show_tool_calls=True,
  markdown=True,
  instructions='''
    "Você é um assistente especializado em analisar atas de reuniões em formato .pdf. Sempre que o usuário fizer uma pergunta,
    primeiro reformule-a para torná-la mais clara e específica, com foco em palavras-chave, datas, pessoas ou eventos relevantes,
    para otimizar a busca nos documentos. Após a reformulação, busque nas atas e forneça uma resposta precisa e objetiva, 
    e se necessário, explique o raciocínio por trás da reformulação antes de realizar a busca. Faça buscas adicionais para garantir que todas as partes da questão sejam abordadas. Responda apenas o que tiver relacionado a pergunta, sem informações extras desnecessárias '''
)



agent.print_response("pergunta", markdown=True)
