pip install pyswarm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import sentiwordnet as swn
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pyswarm import pso


nltk.download('sentiwordnet')
nltk.download('wordnet')


df=pd.read_csv('/content/sample30.csv')
df.head()


def sentiment_amplification(text):
    words = text.split()
    sentiment_score = 0
    for word in words:
        synsets = list(swn.senti_synsets(word))
        if synsets:
            sentiment_score += synsets[0].pos_score() - synsets[0].neg_score()

    return sentiment_score

print(sentiment_amplification("Good"))



def get_bert_embeddings(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=512)
        # Get the embeddings from BERT (shape: [batch_size, seq_length, embedding_dim])
        embedding = model(**inputs).last_hidden_state
        # Average the embeddings across all tokens (mean pooling)
        avg_embedding = tf.reduce_mean(embedding, axis=1).numpy()
        embeddings.append(avg_embedding)

    return np.array(embeddings)
print(get_bert_embeddings('i love this products'))



# Encode Sentiments using LabelEncoder
le = LabelEncoder()
df['Sentiment'] = le.fit_transform(df['Sentiment'])
print(df['Sentiment'])


df['sentiment_score'] = df['Review'].apply(sentiment_amplification)
X_bert_embeddings = get_bert_embeddings(df['Review'])
print(X_bert_embeddings)



y = pd.get_dummies(df['Sentiment'])
print(y)


X_train, X_test, y_train, y_test = train_test_split(X_bert_embeddings, y, test_size=0.2, random_state=42)
X_train = tf.squeeze(X_train, axis=1)
# Check Shapes of X_train and y_train
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


def build_model(embedding_dim, learning_rate, num_units1, num_units2):
    """
    Function to build the model with dynamic hyperparameters.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(embedding_dim,)),
        tf.keras.layers.Dense(num_units1, activation='relu'),
        tf.keras.layers.Dense(num_units2, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model




def objective_function(params):
    """
    Objective function for PSO to minimize (in this case, validation loss).
    `params` contains hyperparameters: learning_rate, num_units1, num_units2
    """
    learning_rate, num_units1, num_units2 = params
    
    # Build the model with current hyperparameters
    model = build_model(embedding_dim=768, learning_rate=learning_rate, 
                        num_units1=int(num_units1), num_units2=int(num_units2))

    # Train the model (use validation data for evaluation)
    model.fit(X_train, y_train, epochs=5, batch_size=2, validation_data=(X_train, y_train), verbose=0)
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(X_train, y_train, verbose=0)
    
    # The PSO tries to minimize the function, so return the validation loss
    return val_loss

# Define the lower and upper bounds for PSO (e.g., learning rate, number of units)
lower_bounds = [0.0001, 32, 32]  # learning rate, num_units1, num_units2
upper_bounds = [0.01, 256, 256]  # learning rate, num_units1, num_units2

# Run the PSO algorithm
best_params, best_value = pso(objective_function, lower_bounds, upper_bounds, swarmsize=10, maxiter=5)

# Extract the best parameters
best_learning_rate, best_num_units1, best_num_units2 = best_params

# Train the model with the best parameters
model = build_model(embedding_dim=768, learning_rate=best_learning_rate, 
                    num_units1=int(best_num_units1), num_units2=int(best_num_units2))

# Final model training with best hyperparameters
history=model.fit(X_train, y_train, epochs=5, batch_size=2, validation_data=(X_train, y_train))
print(history)



epochs = range(1, len(history.history['loss']) + 1)
# Plot Loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.title('Final Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Final Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
