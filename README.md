This is a RAG application used on open source models like Deepseek, LLama and others using Ollama.

Addressed painpoint: all the available RAG apps are limited to the less than 500 pages. either they get stuck by loading process or others issues. this files address that pain point. this will definitly helps you to process morethan 500 pages and I have tried with the AWS glu pdf document.
Advantage of using of this approach is we have control on our data

Prerequisites: You need to install and run ollama locally. Go to command prompt and run ollam run <model_name> #model in this file used deepseek-r1:1.5b

streamlit run <file_name>

example: streamlit run ollama_RAG.py

You can start chatting with the pdf.

Happy learning and coding.
