# ATAS - Analisador de Atas

ATAS é uma aplicação para análise de atas de reuniões usando inteligência artificial.

## Requisitos

- Python 3.8+
- Chave de API do OpenAI (configure no arquivo `.env`)

## Instalação

1. Clone este repositório
2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
3. Crie um arquivo `.env` na raiz do projeto com sua chave de API:
   ```
   OPENAI_API_KEY=sua_chave_aqui
   ```

## Uso

1. Execute a aplicação:

   ```
   python run_app.py
   ```

2. Adicione seus arquivos PDF ou TXT na pasta `docs/` (criada automaticamente na primeira execução)

3. Clique em "Carregar Documentos" para indexar os arquivos

4. Digite suas perguntas sobre as atas na caixa de texto

5. Clique em "Perguntar" para obter respostas baseadas no conteúdo das atas

## Funcionalidades

- Processamento de documentos PDF e TXT
- Análise semântica do conteúdo usando IA
- Interface gráfica simples e intuitiva
- Persistência de dados entre sessões

## Solução de Problemas

- Se os documentos não carregarem, verifique se estão no formato PDF ou TXT
- Para problemas com respostas, verifique sua chave de API do OpenAI
- Use o botão "Atualizar Índice" para reindexar documentos quando necessário

## Requisitos de Sistema

- Mínimo de 4GB de RAM recomendado
- O processamento de documentos grandes pode consumir mais memória
