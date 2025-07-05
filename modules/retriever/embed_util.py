import contriever.src.contriever
import numpy as np
import os
import torch

from gritlm import GritLM
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from ..corpus import Corpus

def load_embed_model(method, device, mode="default"):
    if mode == "grit":
        return GritLM(method, torch_dtype="auto", mode="embedding"), None
    elif mode == "reasonIR":
        model = AutoModel.from_pretrained(method, torch_dtype="auto", trust_remote_code=True)
        model.eval()
        return model.to(device)
    elif mode == "contriever":
        # TODO: compare if this is any different than just using HF auto wrappers
        model, tokenizer, _ = contriever.src.contriever.load_retriever(method)
        return model, tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(method)
    model = AutoModel.from_pretrained(method)

    return model.to(device), tokenizer

def embed_passages(corpus: Corpus, config):
    # Embed config args
    embed_dir = config.embed_dir
    embed_model = config.embed_model
    embed_size = config.embed_size
    batch_size = config.batch_size
    device = config.device

    # Output for embeddings
    os.makedirs(embed_dir, exist_ok=True)
    embed_file = os.path.join(embed_dir, f"{corpus.get_name()}_{len(corpus)}_{embed_model.split('/')[-1]}.npy")

    model, tokenizer = load_embed_model(embed_model)
    model = model.to(device)

    if embed_size:
        output_shape = (len(corpus), embed_size)
        embeddings_memmap = np.memmap(embed_file, dtype='float32', mode='w+', shape=output_shape)
    else:
        # Get shape from embedding
        output_shape = None
        embeddings_memmap = None
    
    with torch.no_grad():
        # Process passages in batches
        print(len(corpus))
        for i in tqdm(range(0, len(corpus), batch_size)):
            # TODO: maybe need additional processing
            # TODO: WHhen corpus is sharded and not fully loaded in mem,
            # indexing should dynamically load shards
            batch = [datum["text"] for datum in corpus[i:i + batch_size]]
            
            embeddings = embed_batch(batch, model, tokenizer, device)

            if embeddings_memmap is None:
                # Instantiate embeddings map with length of corpus and shape of embedding
                print(embeddings.shape)
                embed_shape = embeddings.shape
                output_shape = (len(corpus), embed_shape)
                embeddings_memmap = np.memmap(embed_file, dtype='float32', mode='w+', shape=output_shape)

            # Move embeddings to CPU and convert to numpy
            embeddings_memmap[i:i + len(batch)] = embeddings
            embeddings_memmap.flush()

            write_equal = np.all(np.equal(embeddings[0], embeddings_memmap[i]))

            assert write_equal

        print(embeddings_memmap.shape)

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def embed_batch(batch, model, tokenizer, device, mode="default"):
    with torch.no_grad():
        if mode == "grit":
            print("using grit")
            def gritlm_instruction(instruction):
                return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

            embeddings = model.encode(batch, batch_size=64, instruction=gritlm_instruction(""))
        elif mode == "reasonIR":
            print("using reasonIR")
            instruction = ""
            embeddings = model.encode(batch, batch_size=32, instruction=instruction)
        elif mode == "contriever":
            assert device == "cuda", "Contriever only supports GPU"
            model.eval()
            model.to(device)
            embeddings, batch_question = [], []
            with torch.no_grad():
                for k, q in tqdm(enumerate(batch)):
                    batch_question.append(q)

                    if len(batch_question) == 512 or k == len(batch) - 1:
                        encoded_batch = tokenizer.batch_encode_plus(
                            batch_question,
                            return_tensors="pt",
                            max_length=512,
                            padding=True,
                            truncation=True,
                        )
                        encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
                        assert all(v.device.type == "cuda" and v.device.index == 0 for v in encoded_batch.values()), \
                        f"Some tensors are not on cuda:0: {[v.device for v in encoded_batch.values()]}"

                        output = model(**encoded_batch)
                        embeddings.append(output.cpu())

                        batch_question = []
            embeddings = torch.cat(embeddings, dim=0).numpy()
        else:
            # Tokenize the batch
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

            # Generate embeddings
            # TODO: Handle embedding extraction for embedders other than contriever
            # Mean pooling over token embeddings
            outputs = model(**inputs)#.last_hidden_state.mean(dim=1)  
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

            # Move embeddings to CPU and convert to numpy
            embeddings = embeddings.cpu().numpy()

    assert not np.isnan(embeddings).any()
    return embeddings
