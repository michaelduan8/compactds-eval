import argparse
import contriever.src.normalize_text
import os
import re

from tqdm import tqdm 

from src.utils import load_jsonl, write



def check_below_lexical_overlap_threshold(doc, gold_text, threshold=0.25, mode='longest'):
    """
    Check if the *doc* has no more than *threshold* overlapping with the *gold_text*.
    If yes, return True; else return False.

    *threshold* is set between [0,1] which defines the ratio of tokens that are ok to overlap.
    *mode*: choose from ['longest', 'jaccard']
    """
    if threshold == 1:
        return True
    
    if mode == 'longest':
        doc_words = doc.split(' ')
        gold_text_words = gold_text.split(' ')

        max_overlap = max_contiguous_overlap(doc_words, gold_text_words)
        
        if threshold < 1:
            # print(max_overlap, len(gold_text_words) * threshold, max_overlap < int(len(gold_text_words) * threshold))
            return max_overlap < int(len(gold_text_words) * threshold)
        else:
            # when threshold is the word count
            # print(max_overlap, threshold, max_overlap < threshold)
            return max_overlap < threshold
    
    elif mode == 'jaccard':
        assert threshold < 1, f"Jaccard similarity decontamination doesn't support word limit. Set threshold within [0, 1]"
        return check_13word_jaccard_similarity(doc, gold_text, threshold)


def max_contiguous_overlap(list1, list2):
    # Function to find the length of the maximum contiguous overlap
    def find_overlap(start1, start2, list1, list2):
        length = 0
        while start1 + length < len(list1) and start2 + length < len(list2) and list1[start1 + length] == list2[start2 + length]:
            length += 1
        return length

    max_overlap = 0
    for i in range(len(list1)):
        for j in range(len(list2)):
            if list1[i] == list2[j]:
                overlap = find_overlap(i, j, list1, list2)
                max_overlap = max(max_overlap, overlap)

    return max_overlap



# N-gram Jaccard similarity overlap
def generate_13word_grams(text):
    # Split text into words
    words = text.split()
    # Generate all possible sequences of 13 words
    return {' '.join(words[i:i+13]) for i in range(len(words) - 12)}

def jaccard_similarity(set1, set2):
    # Calculate Jaccard similarity between two sets
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

def check_13word_jaccard_similarity(text1, text2, threshold=0.8):
    # Generate 13-word grams for each text
    grams1 = generate_13word_grams(text1)
    grams2 = generate_13word_grams(text2)
    
    # Calculate Jaccard similarity
    similarity = jaccard_similarity(grams1, grams2)
    # print(similarity)
    
    # Return the similarity if it's 0.8 or less, otherwise indicate high overlap
    if similarity > threshold:
        return False
    else:
        return True
    
def str_normalize(text):
    text = text.lower()

    # Replace all whitespace with " "
    text = re.sub(r'\s+', ' ', text)

    text = contriever.src.normalize_text.normalize(text)

    return text

def main(args):
    retrieval_results_paths = args.retrieval_results_paths
    output_dir = args.output_dir

    dedupe_key = args.dedupe_key

    paragraph_delimiter = args.paragraph_delimiter
    normalize = args.normalize

    jaccard_threshold = args.jaccard_threshold
    ls_threshold = args.ls_threshold

    for path in tqdm(retrieval_results_paths, desc="Decontaminating retrieval results"):
        total = load_jsonl(path)
        decontaminated_total = []
        for dt in tqdm(total, desc="Decontaminating for path"):
            query = dt[dedupe_key]
            ctxs = dt["ctxs"]

            if normalize:
                query = str_normalize(query)
            
            decontaminated_ctxs = []
            for ctx in ctxs:
                text = ctx["retrieval text"]
                paragraphs = text.split(paragraph_delimiter)
                decontaminated_paragraphs = []
                for paragraph in paragraphs:
                    if not paragraph:
                        decontaminated_paragraphs.append("")
                        continue
                    
                    original_paragraph = paragraph
                    if normalize:
                        paragraph = str_normalize(paragraph)

                    jaccard_check = check_below_lexical_overlap_threshold(
                        paragraph, 
                        query, 
                        threshold=jaccard_threshold, 
                        mode='jaccard')
                    
                    ls_check = check_below_lexical_overlap_threshold(
                        paragraph, 
                        query, 
                        threshold=ls_threshold, 
                        mode='longest')
                    
                    if not (jaccard_check and ls_check):
                        decontaminated_paragraphs.append("")
                    else:
                        decontaminated_paragraphs.append(original_paragraph)
                
                decontaminated_text = paragraph_delimiter.join(decontaminated_paragraphs)
                decontaminated_ctx = ctx | {"retrieval text": decontaminated_text}
                decontaminated_ctxs.append(decontaminated_ctx)

            decontaminated_total.append(dt | {
                "ctxs": decontaminated_ctxs
            })

        filename = os.path.basename(path)
        filename_wo_extension, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{filename_wo_extension}_decontaminated.jsonl")
        write(output_path, decontaminated_total)


if __name__ == '__main__':
    # Perform paragraph level decontamination
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_results_paths", type=str, nargs="+", help="Files with retrieval results")
    parser.add_argument("--dedupe_key", type=str, default="query", help="Key to decontaminate against")
    parser.add_argument("--output_dir", type=str, default="deduped_retrieval_results", help="Directory for output files")
    parser.add_argument("--paragraph_delimiter", type=str, default="\n", help="delimiter for splitting text")
    parser.add_argument("--normalize", action='store_true', help='Whether to normalize text before performing decontamination check')
    parser.add_argument("--jaccard_threshold", default=0.7)
    parser.add_argument("--ls_threshold", default=13)
    args = parser.parse_args()
    main(args)
    