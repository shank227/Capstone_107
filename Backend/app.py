from flask import Flask, request, jsonify
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from flask_cors import CORS
import numpy as np

class GCNClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=2):  # ✅ 384 instead of 768
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Classification layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # ✅ Dropout again
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # ✅ Apply dropout after activation

        # Aggregate at the graph level
        x = global_mean_pool(x, batch)  # ✅ Pooling for graph classification

        # Final classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)  # ✅ Log softmax for classification

# Load the trained model
model_for_testing = GCNClassifier(input_dim=384, hidden_dim=128, output_dim=2)
model_path = "C:/Users/Public/Studies/For college/Capstone/coding/g_frontendchanged/Backend/gcn_email_classifier.pth"
model_for_testing.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  # Load weights
model_for_testing.eval()

# Load the function to convert text into a PyG graph
from graph_module import create_word_graph, clean_text  # Update with actual module

app = Flask(__name__)
CORS(app, resources={r"/classify": {"origins": ["http://localhost:5173", "http://localhost:3000"]}})

def get_top_contributing_words(word_importance, top_k=10):
    """Get top K words that contributed most to the prediction."""
    sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
    return [{"word": word, "importance": importance} for word, importance in sorted_words[:top_k]]

@app.route("/classify", methods=["POST"])
def classify_email():
    data = request.json
    email_text = data.get("email", "")

    if not email_text:
        return jsonify({"error": "No email text provided"}), 400

    # Get cleaned words for explanation
    words = clean_text(email_text)
    
    # Convert text to a PyG graph
    email_graph = create_word_graph(email_text, window_size=3)
    if email_graph is None or email_graph.x.shape[0] == 0:
        return jsonify({"error": "Could not process email text"}), 400
    
    email_graph = email_graph.to("cpu")

    # Make prediction with gradients enabled for explanation
    # Keep model in eval mode but enable gradients on input
    email_graph.x.requires_grad = True
    
    batch = torch.tensor([0] * email_graph.x.shape[0])
    
    # Forward pass
    output = model_for_testing(email_graph.x, email_graph.edge_index, batch)
    
    # Get probabilities (convert from log_softmax)
    probabilities = torch.exp(output).detach().cpu().numpy()[0]
    prediction = torch.argmax(output, dim=1).item()
    
    # Get confidence score
    confidence = float(probabilities[prediction])
    
    # Get explanation - compute gradients
    score = output[0, prediction]
    score.backward()
    
    # Get gradients w.r.t. input features
    gradients = email_graph.x.grad
    
    if gradients is not None:
        # Compute node importance as L2 norm of gradients
        node_importance = torch.norm(gradients, dim=1).detach().cpu().numpy()
        
        # Normalize importance scores
        if node_importance.max() > node_importance.min():
            node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min())
        else:
            node_importance = np.zeros_like(node_importance)
        
        # Create word-importance mapping
        word_importance = {}
        for i, word in enumerate(words):
            if i < len(node_importance):
                word_importance[word] = float(node_importance[i])
            else:
                word_importance[word] = 0.0
    else:
        # Fallback: return uniform importance if gradients failed
        word_importance = {word: 0.0 for word in words}
    
    top_words = get_top_contributing_words(word_importance, top_k=10)

    label_map = {0: "Legitimate Email", 1: "Phishing Email"}
    
    return jsonify({
        "prediction": label_map[prediction],
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "legitimate": round(float(probabilities[0]) * 100, 2),
            "phishing": round(float(probabilities[1]) * 100, 2)
        },
        "word_importance": word_importance,
        "top_contributing_words": top_words
    })

def load_embeddings(path="C:/Users/Public/Studies/For college/Capstone/coding/g_frontendchanged/word_embeddings.pkl"):
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

if __name__ == "__main__":
    app.run(debug=True)