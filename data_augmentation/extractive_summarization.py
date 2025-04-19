import json
import jsonlines
import spacy
import argparse

def get_max_depth(token):
    if not list(token.children):
        return 0
    else:
        return 1 + max(get_max_depth(child) for child in token.children)

def prune_tree(token, max_depth):
    if max_depth == 0:
        return []
    else:
        subtree = [token]
        for child in token.children:
            subtree.extend(prune_tree(child, max_depth - 1))
        return subtree

def extractive_summarization(input_file, depth_ratio, group_tokens, output_file):
    nlp = spacy.load("en_core_web_sm")
    final_output = []
    
    with open(input_file, 'r') as f:
       data = [json.loads(line) for line in f]
    
    for item in data:
        input_text = item['input_text']
        sentences = input_text.split('. ')
        summarized_sentences = []

        for sentence in sentences:
            doc = nlp(sentence)
            subtree = set()  # Using a set to avoid duplicates
            
            for token in doc:
                if token.dep_ in ['nsubj', 'ROOT', 'dobj']:
                    max_depth = get_max_depth(token)
                    prune_depth = int(max_depth * depth_ratio)
                    pruned_subtree = prune_tree(token, prune_depth)
                    
                    subtree.update(pruned_subtree)
                    if group_tokens:
                        subtree.update([child for child in token.children])
                
            # Sort tokens by their original order
            subtree = sorted(subtree, key=lambda x: x.i)
            
            # Remove prepositions at the end
            while subtree and subtree[-1].pos_ == 'ADP':
                subtree.pop()
            
            summarized_sentence = ' '.join([token.text for token in subtree])
            summarized_sentences.append(summarized_sentence)
        
        final_summary = '. '.join(summarized_sentences)
        final_output.append({
                "input_text": input_text,
                "pseudo_summary_extractive": final_summary
            })
    
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(final_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extractive Summarization')
    parser.add_argument('--input_file', type=str, help='Path to input text file in JSON')
    parser.add_argument('--depth_ratio', type=float, help='Ratio of tree depth to prune')
    parser.add_argument('--group_tokens', action='store_true', help='Boolean flag to group some token nodes before pruning')
    parser.add_argument('--output_file', type=str, help='Path to output file')
    
    args = parser.parse_args()
    
    extractive_summarization(args.input_file, args.depth_ratio, args.group_tokens, args.output_file)