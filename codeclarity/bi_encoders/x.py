from encoder import CodeEmbedder

if __name__ == "__main__": 
    embedder = CodeEmbedder("microsoft/unixcoder-base")
    print(embedder.encode("foo", silence_progress_bar= True, return_generation_metadata= True))