## Documentation

### Integrate Langchain Dataloading
1. Convert the local pretraining docs into langchain Documents
2. Integrate the embedding model from huggingface into langchain
    - Chunk the docs using langchain + tokenizer of model. 
    - Dedup the docs
    - Implement Pydantic
    - Try to do some baseline retrievals. 
    
3. Profile the speed to find bottlenecks to optimize
4. Do Similartiy search using langchain retriever and get the top documents

