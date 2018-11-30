import torch
import torch.nn as nn
import torchvision.models as models



class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        pretrained = models.resnet50(pretrained=True)
        for param in pretrained.parameters():
            param.requires_grad_(False)
        
        modules = list(pretrained.children())[:-1]
        self.pretrained = nn.Sequential(*modules)
        self.embed = nn.Linear(pretrained.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        

    def forward(self, images):
        features = self.pretrained(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1,batch_size=10):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = self.zero_hidden()
        
    def zero_hidden(self):
        return (torch.zeros((1, self.batch_size, self.hidden_size), device=self.device), \
                torch.zeros((1, self.batch_size, self.hidden_size), device=self.device))
    
    def forward(self, features, captions):
        embeds = self.word_embeddings(captions[:,:-1])
        feats = features.unsqueeze(1)
        embeddings = torch.cat((feats,embeds),1)
        
        lstm_out, self.hidden = self.lstm(embeddings)
        
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        embeds = inputs.clone()
        for i in range(max_len):
            outputs, states = self.lstm(embeds, states)
            outputs = self.linear(outputs.squeeze(1))
            ix = outputs.max(1)[1]
            sentence.append(ix.item())
            embeds = self.word_embeddings(ix).unsqueeze(1)
        return sentence