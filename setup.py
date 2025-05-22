from setuptools import setup, find_packages

setup(
    name="iATAS",
    version="1.2.0",
    description="Analisador de ATAS - Document analysis tool for meeting minutes",
    author="iATAS Team",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.22.0",
        "scikit-learn>=1.0.0",
        "python-dotenv>=0.19.0",
        "PyPDF2>=3.0.0",
        "openai>=1.0.0",
        "agno>=0.2.0",
        "PyQt6>=6.2.0",
        "psutil>=5.9.0",
    ],
    entry_points={
        'console_scripts': [
            'iatas=run_app:main',
        ],
    },
) 