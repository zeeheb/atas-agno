# Guia do Usuário - iATAS (Analisador de ATAS)

## Introdução

Bem-vindo ao iATAS, uma ferramenta de análise de documentos especializada em atas de reuniões. Este aplicativo utiliza inteligência artificial para ajudar você a extrair informações relevantes e responder perguntas sobre o conteúdo dos seus documentos.

## Instalação

### Windows (com instalador)

1. Execute o arquivo `iATAS_Setup.exe`
2. Siga as instruções na tela
3. Após a instalação, o programa estará disponível no Menu Iniciar e na Área de Trabalho

### Qualquer Sistema Operacional (sem instalador)

1. Extraia o arquivo zip para uma pasta de sua escolha
2. Execute o arquivo `iATAS.exe` (Windows) ou `iATAS` (Linux/Mac)

## Configuração Inicial

Na primeira execução, você precisará:

1. Configurar uma chave de API da OpenAI:

   - Clique no botão "Configurações" no canto superior direito
   - Insira sua chave de API da OpenAI
   - Clique em "Salvar"

2. Selecionar uma pasta para documentos:
   - Na tela inicial, selecione a pasta onde estão seus documentos
   - Ou coloque seus documentos na pasta "docs" dentro da instalação

## Usando o Aplicativo

### Carregando Documentos

1. Clique no botão "Carregar Documentos"
2. Selecione a pasta que contém seus arquivos PDF ou TXT
3. Aguarde o processamento dos documentos

### Fazendo Perguntas

1. Digite sua pergunta na caixa de texto no painel inferior
2. Clique em "Enviar" ou pressione Enter
3. Aguarde a resposta que será exibida no painel de resultados

### Recursos Adicionais

- **Histórico de Perguntas**: Suas perguntas e respostas anteriores são salvas automaticamente
- **Limpar Histórico**: Use o botão "Limpar Histórico" para apagar o histórico de perguntas
- **Limpar Documentos**: Use o botão "Limpar Documentos" para remover todos os documentos carregados

## Tipos de Documentos Suportados

- Arquivos PDF (.pdf)
- Arquivos de texto (.txt)

## Perguntas Frequentes

### O aplicativo precisa de conexão com a internet?

Sim, o iATAS utiliza a API da OpenAI para processar as perguntas, portanto, é necessária uma conexão com a internet.

### Como obtenho uma chave de API da OpenAI?

Visite [platform.openai.com](https://platform.openai.com/) para criar uma conta e obter sua chave de API.

### Meus documentos são enviados para a OpenAI?

Não, seus documentos são processados localmente. Apenas as consultas específicas que você faz são enviadas para a API da OpenAI.

### Posso usar o aplicativo sem uma chave de API?

Não, uma chave de API da OpenAI é necessária para o funcionamento do aplicativo.

### Como devo formular minhas perguntas?

Seja específico e claro. Por exemplo:

- "Quais foram as principais decisões na reunião de 15 de março?"
- "Quem estava presente na reunião do conselho de administração?"
- "Qual foi o resultado da votação sobre o orçamento?"

## Solução de Problemas

### O aplicativo está lento

- Reduza a quantidade de documentos carregados
- Verifique se seu computador atende aos requisitos mínimos
- Feche outros aplicativos que consomem muita memória

### Erro ao carregar documentos

- Verifique se os arquivos estão nos formatos suportados (PDF, TXT)
- Certifique-se de que os arquivos não estão corrompidos
- Tente dividir grandes quantidades de documentos em carregamentos menores

### Mensagens de erro ao fazer perguntas

- Verifique sua conexão com a internet
- Confirme se sua chave de API da OpenAI é válida
- Verifique se ainda há créditos disponíveis na sua conta da OpenAI

## Suporte

Para obter ajuda adicional, entre em contato com nossa equipe de suporte em:

- Email: suporte@iatas.exemplo.com
- Site: www.iatas.exemplo.com/suporte
