import io
from setuptools import setup, find_packages

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='codeclarity',
    packages=['codeclarity'],
    version='0.1',
    license='MIT',
    description='An embedding tool for all state of the art code language models',
    author='Charlie Masters',
    author_email='cm2435@bath.ac.uk',
    url='https://github.com/DocumaticAI/CodeClarity',
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords=['sentence', 'embedding', 'code', 'search', 'contrastive', 'nlp', 'deep_learning', 'semantic'],
    install_requires=[
    "transformers",
    "accelerate",
    "tqdm",
    "numpy",
    ]
)