from sklearn import metrics
from itertools import chain
from six.moves import range, reduce
import numpy as np
import tensorflow as tf

from data_utils import tokenize, parse_dialogs_per_response
from memory_network import MemoryNetwork

def vectorize_candidates(candidates, word_idx, sentence_size):
    # Determine shape of final vector
    shape = (len(candidates), sentence_size)
    candidates_vector = []
    for i, candidate in enumerate(candidates):
        # Determine zero padding
        zero_padding = max(0, sentence_size - len(candidate))
        # Append to final vector
        candidates_vector.append(
            [word_idx[w] if w in word_idx else 0 for w in candidate] 
            + [0] * zero_padding)
    # Return as TensorFlow constant
    return tf.constant(candidates_vector, shape=shape)

def vectorize_data(data, word_idx, sentence_size, batch_size, max_memory_size):
    facts_vector = []
    questions_vector = []
    answers_vector = []
    # Sort data in descending order by number of facts
    data.sort(key=lambda x: len(x[0]), reverse=True)
    for i, (fact, question, answer) in enumerate(data):
        # Find memory size
        if i % batch_size == 0:
            memory_size = max(1, min(max_memory_size, len(fact)))
        # Build fact vector
        fact_vector = []
        for i, sentence in enumerate(fact, 1):
            fact_padding = max(0, sentence_size - len(sentence))
            fact_vector.append(
                [word_idx[w] if w in word_idx else 0 for w in sentence] 
                + [0] * fact_padding)
        # Keep the most recent sentences that fit in memory
        fact_vector = fact_vector[::-1][:memory_size][::-1]
        # Pad to memory_size
        memory_padding = max(0, memory_size - len(fact_vector))
        for _ in range(memory_padding):
            fact_vector.append([0] * sentence_size)
        # Build question vector
        question_padding = max(0, sentence_size - len(question))
        question_vector = [word_idx[w] if w in word_idx else 0 
                           for w in question] \
                           + [0] * question_padding
        # Append to final vectors
        facts_vector.append(np.array(fact_vector))
        questions_vector.append(np.array(question_vector))
        # Answer is already an integer corresponding to a candidate
        answers_vector.append(np.array(answer))
    return facts_vector, questions_vector, answers_vector

class ChatBotWrapper(object):
    def __init__(self, train_data, test_data, val_data, 
                 candidates, candidates_to_idx,
                 memory_size, batch_size, learning_rate, 
                 evaluation_interval, hops,
                 epochs, embedding_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.evaluation_interval = evaluation_interval
        self.epochs = epochs

        self.candidates = candidates 
        self.candidates_to_idx = candidates_to_idx
        self.candidates_size = len(candidates)
        self.idx_to_candidates = dict((self.candidates_to_idx[key], key) 
                                      for key in self.candidates_to_idx)
        # Initialize data and build vocabulary
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.build_vocab(train_data + test_data + val_data, candidates)
        # Vectorize candidates
        self.candidates_vec = vectorize_candidates(
            candidates, self.word_idx, self.candidate_sentence_size)
        # Initialize optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # Initialize TensorFlow session and Memory Network model
        self.sess = tf.Session()
        self.model = MemoryNetwork(
                        self.sentence_size, self.vocab_size, 
                        self.candidates_size, self.candidates_vec, 
                        embedding_size, hops,
                        optimizer=optimizer, session=self.sess)
    
    def build_vocab(self, data, candidates):
        # Build word vocabulary set from all data and candidate words
        vocab = reduce(lambda x1, x2: x1 | x2, 
            (set(list(chain.from_iterable(facts)) + questions) 
                for facts, questions, answers in data))
        vocab |= reduce(lambda x1, x2: x1 | x2, 
            (set(candidate) for candidate in candidates))
        vocab = sorted(vocab)
        # Assign integer indices to each word
        self.word_idx = dict((word, idx + 1) for idx, word in enumerate(vocab))
        # Compute various data size numbers
        max_facts_size = max(map(len, (facts for facts, _, _ in data)))
        self.sentence_size = max(
            map(len, chain.from_iterable(facts for facts, _, _ in data)))
        self.candidate_sentence_size = max(map(len, candidates))
        question_size = max(map(len, (questions for _, questions, _ in data)))
        self.memory_size = min(self.memory_size, max_facts_size)
        self.vocab_size = len(self.word_idx) + 1  # +1 for null word
        self.sentence_size = max(question_size, self.sentence_size)
        
    def predict_for_batch(self, facts, questions):
        preds = []
        # Iterate over mini-batches
        for start in range(0, len(facts), self.batch_size):
            end = start + self.batch_size
            facts_batch = facts[start:end]
            questions_batch = questions[start:end]
            # Predict per batch
            pred = self.model.predict(facts_batch, questions_batch)
            preds += list(pred)
        return preds    

    def train(self):
        # Vectorize training and validation data
        train_facts, train_questions, train_answers = vectorize_data(
            self.train_data, self.word_idx, self.sentence_size, 
            self.batch_size, self.memory_size)
        val_facts, val_questions, val_answers = vectorize_data(
            self.val_data, self.word_idx, self.sentence_size, 
            self.batch_size, self.memory_size)
        # Chunk training data into batches
        batches = zip(range(0, len(train_facts) - self.batch_size, 
                            self.batch_size),
                      range(self.batch_size, len(train_facts), 
                            self.batch_size))
        batches = [(start, end) for start, end in batches]
        # Start training loop
        for epoch in range(1, self.epochs + 1):
            np.random.shuffle(batches)
            total_cost = 0.0
            for start, end in batches:
                facts = train_facts[start:end]
                questions = train_questions[start:end]
                answers = train_answers[start:end]
                # Train on batch
                batch_cost = self.model.fit(facts, questions, answers)
                total_cost += batch_cost
            if epoch % self.evaluation_interval == 0:
                # Compute accuracy over training and validation set
                train_preds = self.predict_for_batch(
                    train_facts, train_questions)
                val_preds = self.predict_for_batch(
                    val_facts, val_questions)
                train_acc = metrics.accuracy_score(
                    train_preds, train_answers)
                val_acc = metrics.accuracy_score(
                    val_preds, val_answers)
                print("Epoch: ", epoch)
                print("Total Cost: ", total_cost)
                print("Training Accuracy: ", train_acc)
                print("Validation Accuracy: ", val_acc)
                print("---")
    
    def test(self):
        # Compute accuracy over test set
        test_facts, test_questions, test_answers = vectorize_data(
            self.test_data, self.word_idx, self.sentence_size, 
            self.batch_size, self.memory_size)
        test_preds = self.predict_for_batch(test_facts, test_questions)
        test_acc = metrics.accuracy_score(test_preds, test_answers)
        print("Testing Accuracy: ", test_acc)
        
    def interactive_mode(self):
        facts = []
        utterance = None
        response = None
        turn_count = 1
        while True:
            line = input("==> ").strip().lower()
            if line == "exit":
                break
            if line == "restart":
                facts = []
                turn_count = 1
                print("Restarting dialog...\n")
                continue
            utterance = tokenize(line)
            data = [(facts, utterance, -1)]
            # Vectorize data and make prediction
            f, q, a = vectorize_data(data, self.word_idx, 
                self.sentence_size, self.batch_size, self.memory_size)
            preds = self.model.predict(f, q)
            response = self.idx_to_candidates[preds[0]]
            # Print predicted response
            print(response)
            response = tokenize(response)
            # Add turn count temporal encoding
            utterance.append("$u")
            response.append("$r")
            # Add utterance/response encoding
            utterance.append("#" + str(turn_count))
            response.append("#" + str(turn_count))
            # Update facts memory
            facts.append(utterance)
            facts.append(response)
            turn_count += 1

if __name__ == "__main__":
    candidates = []
    candidates_to_idx = {}
    with open('dialog-babi/dialog-babi-candidates.txt') as f:
        for i, line in enumerate(f):
            candidates_to_idx[line.strip().split(' ', 1)[1]] = i
            line = tokenize(line.strip())[1:]
            candidates.append(line)

    train_data = []
    with open('dialog-babi/dialog-babi-task5-full-dialogs-trn.txt') as f:
        train_data = parse_dialogs_per_response(f.readlines(), candidates_to_idx)

    test_data = []
    with open('dialog-babi/dialog-babi-task5-full-dialogs-tst.txt') as f:
        test_data = parse_dialogs_per_response(f.readlines(), candidates_to_idx)

    val_data = [] 
    with open('dialog-babi/dialog-babi-task5-full-dialogs-dev.txt') as f:
        val_data = parse_dialogs_per_response(f.readlines(), candidates_to_idx)

    chatbot = ChatBotWrapper(train_data, test_data, val_data,
                             candidates, candidates_to_idx,
                             memory_size=50,
                             batch_size=32,
                             learning_rate=0.001,
                             evaluation_interval=10,
                             hops=3,
                             epochs=200,
                             embedding_size=50)
    chatbot.train()
    chatbot.test()
    chatbot.interactive_mode()
