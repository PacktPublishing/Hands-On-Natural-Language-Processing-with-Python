import re

def tokenize(sent):
    stop_words = {"a", "an", "the"}
    sent = sent.lower()
    if sent == '<silence>':
        return [sent]
    # Convert sentence to tokens
    result = [word.strip() for word in re.split('(\W+)?', sent) 
              if word.strip() and word.strip() not in stop_words]
    # Cleanup
    if not result:
        result = ['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result = result[:-1]
    return result

def parse_dialogs_per_response(lines, candidates_to_idx):
    data = []
    facts_temp = []
    utterance_temp = None
    response_temp = None
    # Parse line by line
    for line in lines:
        line = line.strip()
        if line:
            nid, line = line.split(' ', 1)
            if '\t' in line: # Has utterance and respone
                utterance_temp, response_temp = line.split('\t')
                # Convert answer to integer index
                answer = candidates_to_idx[response_temp]
                # Tokenize sentences
                utterance_temp = tokenize(utterance_temp)
                response_temp = tokenize(response_temp)
                # Add (facts, question, answer) tuple to data
                data.append((facts_temp[:], utterance_temp[:], answer))
                # Add utterance/response encoding
                utterance_temp.append('$u')
                response_temp.append('$r')
                # Add turn count temporal encoding
                utterance_temp.append('#' + nid)
                response_temp.append('#' + nid)
                # Update facts
                facts_temp.append(utterance_temp)
                facts_temp.append(response_temp)
            else: # Has KB Fact
                response_temp = tokenize(line)
                response_temp.append('$r')
                response_temp.append('#' + nid)
                facts_temp.append(response_temp)
        else: # Start of new dialog
            facts_temp = []
    return data