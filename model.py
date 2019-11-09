import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        #create hidden layer size
        self.hidden_size = hidden_size
        #create n_layers
        self.n_layers = num_layers
        #create embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        #create LSTM, batch_first "TRUE" since we are batching
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers = num_layers, batch_first = True)
        # linear layer that maps the hidden state output dimension 
            # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        self.out = nn.Linear(hidden_size, vocab_size)
        #Initialize hidden state        
        #self.hidden = self.init_hidden(64)
        #self.device = device

    def forward(self, features, captions):
        #get embeddings for each batch of captions, size: [batch_size, caption_lenght]
        embeddings = self.word_embeddings(captions[:,:-1]) 
        #concat image features and captions, shape: [batch, inputs, embeddings]
        lstm_inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        #apply lstm the the stacked features and captions
        out_lstm, _ = self.lstm(lstm_inputs)
        #apply the linear layer to the lstm output
        output = self.out(out_lstm)
        
        return output
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        return (torch.zeros((1, batch_size, self.hidden_size), device = device),
                torch.zeros((1, batch_size, self.hidden_size), device = device))
 
    
    def sample(self, inputs, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled = []
        hidden = self.init_hidden(inputs.shape[0]) 
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden)
            linear_output = self.out(lstm_out)  
            linear_output = linear_output.squeeze(1) 
            _, max_prob = torch.max(linear_output, dim=1) 
            #build the caption sentence
            sampled.append(max_prob.item())
            #if the <END> word happens to appear stop the iteration
            if (max_prob == 1):
                break
            #predicted max prob word is going to be the new input
            inputs = self.word_embeddings(max_prob).unsqueeze(1)

        return sampled