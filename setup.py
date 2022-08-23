import io
from setuptools import setup, find_packages

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='codeclarity',
    packages=['CodeClarity'],
    version='0.1',
    license='MIT',
    description='An embedding tool for all state of the art code language models',
    author='Charlie Masters',
    author_email='cm2435@bath.ac.uk',
    url='https://github.com/DocumaticAI/CodeClarity',
    #download_url='https://github.com/princeton-nlp/SimCSE/archive/refs/tags/0.4.tar.gz',
    keywords=['sentence', 'embedding', 'code', 'search', 'contrastive', 'nlp', 'deep_learning', 'semantic'],
    install_requires=[
    "transformers",
    "accelerate",
    "tqdm",
    "numpy",
    ]
)