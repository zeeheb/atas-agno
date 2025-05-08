import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
pasta_entrada = BASE_DIR / "docx"
pasta_saida = BASE_DIR / "pdfs"

def converter_docx_para_pdf(pasta_entrada, pasta_saida):
    pasta_entrada = Path(pasta_entrada)
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    arquivos_docx = list(pasta_entrada.glob("*.docx"))

    for arquivo in arquivos_docx:
        print(f"Convertendo: {arquivo.name}")
        comando = [
            "C:\\Program Files\\LibreOffice\\program\\soffice.exe",
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(pasta_saida),
            str(arquivo)
        ]
        try:
            subprocess.run(comando, check=True)
            print(f"✔️  {arquivo.name} convertido com sucesso.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao converter {arquivo.name}: {e}")

# Exemplo de uso
converter_docx_para_pdf(pasta_entrada,pasta_saida)
