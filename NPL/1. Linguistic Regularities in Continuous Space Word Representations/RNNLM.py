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

"""
import numpy as np

# Define the input and target
input = np.array([0.0692, 0.0647, -0.0035, 0.2791, 0.0798, 0.4039, 0.0070, 0.0169, 0.1067, 0.2132, -0.0658, 0.4914])
target = 4

# Calculate the softmax of the input
exp_input = np.exp(input)
softmax_input = exp_input / np.sum(exp_input)

# Calculate the cross-entropy loss
loss = -np.log(softmax_input[target])
print(loss)
"""