import csv
import json
import pandas as pd
import smart_open

from tqdm import tqdm

class Corpus(object):
    def __init__(self, corpus_config):
                 
                 
        # corpus_name, data_paths, data_size=None, block_length=None, stride=None, sharded=False):
        '''
        Load corpus from data paths 
        '''
        self.corpus_name = corpus_config.corpus_name
        self.data_paths = corpus_config.data_paths
        self.block_length = corpus_config.block_length
        self.stride = corpus_config.stride  # TODO: what is stride?
        self.sharded = corpus_config.sharded
        self.prepend_title = corpus_config.prepend_title

        self.data_size = corpus_config.data_size
        if not self.sharded:
            self.load()
        

    def load(self):
        # TODO: See if other loading beyond smart_open needed
        # Loads corpus from data_paths assuming a format with a text key
        data = []
        idx = 0
        for path in self.data_paths:
            print(f"Loading passages from: {path}")
            with smart_open.open(path) as f:
                # TODO: Try local reading rn; smart_open doesn't seem to serve large files very well
                if path.endswith(".jsonl"):
                    for dt in tqdm(f):
                        datum = json.loads(dt)
                        text = datum["text"]
                        blocks = self.block_data(text)
                        for block in blocks:
                            # TODO: May need to add title
                            formatted_text = f"{datum['title']} {block}" if self.prepend_title else block
                            if idx == 0:
                                print(formatted_text)

                            data.append({
                                "id": idx,
                                "text": formatted_text
                            })
                            idx += 1   

                        if self.data_size and idx >= self.data_size: # TODO: data_size limiting should be post-loading and should be randomly sampled
                            break
                elif "csv" in path or "tsv" in path:
                    delimiter = "," if "csv" in path else "\t"
                    reader = csv.reader(f, delimiter=delimiter)
                    for row in tqdm(reader):
                        if row[0] == "598451":
                            print(row)
                        if row[0]!="id":
                            data.append({"id": row[0], "text": f"{row[2]} {row[1]}"})
                            idx += 1

                        if self.data_size and idx >= self.data_size: # TODO: data_size limiting should be post-loading and should be randomly sampled
                            break
                else:
                    raise NotImplementedError()

        print(data[0])
        self.data = data
    
    def load_shards(self):
        pass
    
    def block_data(self, text, min_block_length=0, keep_last=True):
        if self.block_length is None:
            return [text]
        
        text = text.split()
        N = len(text) if keep_last else len(text) - len(text) % self.block_length
        blocks = [' '.join(text[i:i + self.block_length]) for i in range(0, N, self.block_length)]

        if len(blocks) > 1 and len(blocks[-1].split(' ')) < min_block_length:
            # merge the last min_block_length words to the previous block
            last_chunk = blocks.pop()
            blocks[-1] += ' ' + last_chunk

        return blocks

    def __len__(self):
        return self.data_size if self.data_size else len(self.data)
    
    def get_name(self):
        return self.corpus_name
    
    def get_data(self):
        return self.data[:self.data_size] if self.data_size else self.data
    
    def get_items(self, indices):
        return [self[idx] for idx in indices]

    def __getitem__(self, idx):
        return self.data[idx]

