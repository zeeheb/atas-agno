# iATAS Installer Guide

Este guia explica como criar um instalador para o aplicativo iATAS para distribuição.

## Requisitos

Para criar o instalador, você precisará:

1. Python 3.8 ou superior
2. Todas as dependências listadas em `requirements.txt`
3. PyInstaller (`pip install pyinstaller`)
4. NSIS (Nullsoft Scriptable Install System) - apenas para Windows

## Passos para criar o instalador

### Método 1: Script automatizado (Recomendado)

1. Navegue até a pasta do projeto `atas-agno`
2. Execute o script de construção:

```bash
python build_installer.py
```

3. O script irá:
   - Instalar o PyInstaller se não estiver disponível
   - Criar um executável no diretório `dist/iATAS`
   - Se o NSIS estiver instalado (Windows), criar um instalador `iATAS_Setup.exe`

### Método 2: Manual

#### Criando o executável com PyInstaller

1. Instale o PyInstaller:

```bash
pip install pyinstaller
```

2. Execute o PyInstaller:

```bash
pyinstaller --name="iATAS" --windowed --add-data "app_data;app_data" --add-data "locale;locale" --add-data "docs;docs" run_app.py
```

#### Criando o instalador com NSIS (Windows)

1. Instale o NSIS de [https://nsis.sourceforge.io/Download](https://nsis.sourceforge.io/Download)
2. Use o script `installer.nsi` gerado ou crie seu próprio script NSIS
3. Compile o script:

```bash
"C:\Program Files (x86)\NSIS\makensis.exe" installer.nsi
```

## Distribuição

Para distribuir o aplicativo, você tem duas opções:

1. **Com Instalador** (Windows): Distribua o arquivo `iATAS_Setup.exe` criado pelo NSIS.
2. **Sem Instalador**: Comprima a pasta `dist/iATAS` em um arquivo zip e distribua-o.

### Instruções para usuários finais

Inclua estas instruções com sua distribuição:

#### Com Instalador (Windows)

1. Execute `iATAS_Setup.exe`
2. Siga as instruções do instalador
3. Inicie o aplicativo pelo atalho criado no desktop ou menu iniciar

#### Sem Instalador

1. Extraia o arquivo zip para qualquer pasta
2. Execute `iATAS.exe` para iniciar o aplicativo

## Configuração após instalação

Os usuários devem configurar:

1. Uma chave de API OpenAI em Configurações
2. Adicionar documentos na pasta `docs` para análise

## Solução de problemas

Se o usuário encontrar problemas ao executar o aplicativo:

1. Verifique se todos os arquivos foram extraídos corretamente
2. Verifique se há um arquivo `app.log` para mensagens de erro
3. Certifique-se de que a pasta `app_data` está presente e acessível
4. Em sistemas Windows, pode ser necessário instalar o Visual C++ Redistributable
