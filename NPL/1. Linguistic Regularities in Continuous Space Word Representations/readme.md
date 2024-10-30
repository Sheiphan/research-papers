# Introduction

A defining feature of neural network language models is their representation of words as high dimensional real valued vectors.

Main advantages of using continuous space word representations (CSWRs) in language models compared to traditional n-gram models.

**Here's a breakdown of the key points:**

- **Discrete vs. Continuous:** N-gram models represent words as discrete units, meaning they have no inherent relationships with other words. CSWRs, on the other hand, represent words as vectors in a continuous space. This allows words with similar meanings to have similar vectors, which is not possible with discrete representations.
- **Generalization:** Due to this continuous representation, CSWRs can achieve a level of generalization that n-gram models cannot. When the model adjusts its parameters based on a specific word or sequence, these adjustments also benefit similar words and sequences. This is because similar words will have similar vectors and therefore be influenced by the same updates.

**Examples:**

- Imagine an n-gram model trained on the sentence "The cat sat on the mat." When it encounters the word "dog" in a new sentence, it doesn't know anything about its relationship to "cat," as they are treated as entirely separate entities.
- In contrast, a CSWR would represent "cat" and "dog" as vectors close to each other in the continuous space. This means that when the model learns something about "cat," this knowledge can be applied to "dog" as well, leading to better predictions in similar contexts.

In this work, we find that the learned word representations in fact capture meaningful syntactic and semantic regularities in a very simple way. Specifically, the regularities are observed as constant vector offsets between pairs of words sharing a particular relationship. For example, if we denote the vector for word $i$ as $x_{i}$ , and focus on the singular/plural relation, we observe that $x_{apple}−x_{apples} ≈ x_{car}−x_{cars}, x_{family}−x_{families}$, and so on. Perhaps more surprisingly, we find that this is also the case for a variety of semantic relations, as measured by the SemEval 2012 task of measuring relation similarity.

# Recurrent Neural Network Model

## **Recurrent Neural Network Language Model in PyTorch**

Here's an explanation of the RNN language model described in the paper, along with a basic implementation in PyTorch:

**1. Network Architecture:**

- **Input layer:** This layer receives a one-hot encoded vector `w(t)` representing the current word at time step `t`.
- **Hidden layer:** This layer has recurrent connections, meaning its output at time step `t` depends not only on the current input but also on the hidden state at the previous time step `s(t-1)`. This state `s(t)` captures the context of the sentence history so far.
- **Output layer:** This layer produces a probability distribution over the vocabulary, indicating the likelihood of each word appearing next.

**2. Equations and Activation Functions:**

- **Hidden state update:**
    
    $$
    s(t) = f (Uw(t) + Ws(t−1))
    $$
    
    ```python
    s(t) = torch.tanh(torch.mm(U, w(t)) + torch.mm(W, s(t-1)))
    
    ```
    
    where:
    * `U` and `W` are weight matrices, learned during training.
    * `mm` is the matrix multiplication operation.
    * `tanh` is the hyperbolic tangent activation function.
    
- **Output layer:**
    
    $$
    y(t) = g (Vs(t))
    $$
    
    ```python
    y(t) = torch.softmax(torch.mm(V, s(t)), dim=1)
    
    ```
    
    where:
    * `V` is another weight matrix.
    * `softmax` function is used to normalize the outputs into a probability distribution.
    

**3. Training:**

The model is trained to maximize the log-likelihood of the observed data. This is done using the backpropagation algorithm with gradient descent.

**4. Code Example:**

```python
import torch
import torch.nn as nn

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # Added batch_first=True
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):  # Added hidden state as input
        embedded_x = self.embedding(x)
        output, hidden = self.rnn(embedded_x, hidden)  # Updated to handle hidden state
        output = self.output(output)
        return output, hidden  # Return hidden state

vocab = ["the", "cat", "sat", "on", "mat", 'is', 'a','sitting', 'sit', 'where', 'what', 'doing']
train_data = [
    ("the cat sat on", "mat"),
    ("the cat is on", "mat"),
    ("the mat is where the cat", "sit")
]

model = RNNLM(vocab_size=len(vocab), embedding_dim=128, hidden_dim=256)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(1000):  # Increased the number of epochs for better training
    for input, target in train_data:
        encoded_input = torch.tensor([vocab.index(word) for word in input.split()])
        encoded_target = torch.tensor([vocab.index(target)])  # Encoded target as a tensor

        optimizer.zero_grad()
        hidden = None  # Initialize the hidden state
        for i in range(encoded_input.size(0)):  # Forward pass through the RNN
            output, hidden = model(encoded_input[i].view(1, 1), hidden)
        loss = loss_fn(output.view(1, -1), encoded_target)  # Calculated loss based on the final output
        loss.backward()
        optimizer.step()

# Predict the next word for the example sentence
encoded_sentence = torch.tensor([vocab.index(word) for word in "the cat is sitting on".split()])
hidden = None
for i in range(encoded_sentence.size(0)):  # Forward pass to get the hidden state
    output, hidden = model(encoded_sentence[i].view(1, 1), hidden)
predicted_word_index = torch.argmax(output.view(-1)).item()
predicted_word = vocab[predicted_word_index]

print(f"Predicted next word: {predicted_word}")
```

This code provides a basic implementation of the RNNLM described in the paper. It uses PyTorch modules for embedding, RNN, and linear layers. The model can be further extended by adding additional functionalities, such as dropout or pre-trained word embeddings.

**Note:** This code is a simplified example and may require adjustments for specific datasets and tasks.