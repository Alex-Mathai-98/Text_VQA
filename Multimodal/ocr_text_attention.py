import torch, torch.nn as nn, torch.nn.functional as f
import random, numpy as np

class VectorizedOCRTokensAndText(nn.Module):
    def __init__(self, question_dim: int = 768, ocr_token_dim: int = 50):
        super().__init__()
        self.linear1 = nn.Linear(question_dim, ocr_token_dim)

    def loop_forward(self, question_feat: torch.tensor, ocr_token_feats: torch.tensor, num_tokens: list) -> torch.tensor:
        """
        question_feat:    question feature of size - question_dim
        ocr_token_feats:  tensor of all ocr token embeddings of size - (k, ocr_token_dim)
        """
        # Get the batch_size 
        batch_size, question_dim = question_feat.shape
        loop_forwards = []

        for i in range(batch_size):

            # Question feature needs to match ocr token embedding size
            # Change size with linear layer
            adjusted_question_feat = self.linear1(question_feat[i])

            # Element wise multiply every ocr token with new question feature
            # hadamard.shape = (k, ocr_token_dim)
            hadamard = adjusted_question_feat.unsqueeze(0) * ocr_token_feats[i]

            # Calculate the attention weights ; 
            # pre_weights.shape = (k) 
            pre_weights = torch.sum(hadamard, dim = 1, dtype = float)

            # Apply softmax
            weights = f.softmax(pre_weights, dim = 0)

            # Apply this attention over the ocr tokens
            # attended_feature.shape = (1, ocr_token_dim)
            attended_feature = torch.mm(weights.unsqueeze(0), ocr_token_feats[i].type(torch.float64))
            
            # Final output returns shape (ocr_token_dim)
            loop_forwards.append(attended_feature.squeeze())

        return torch.stack(loop_forwards)

    def forward(self, question_feat: torch.tensor, ocr_token_feats: torch.tensor, num_tokens: list) -> torch.tensor:
        """
        question_feat:    question feature of size - (batch_size, question_dim)
        ocr_token_feats:  tensor of all ocr token embeddings of size - (batch_size, max_tokens, ocr_token_dim)
        """
        # Get the batch_size 
        batch_size, question_dim = question_feat.shape

        # Question feature needs to match ocr token embedding size
        # Change size with linear layer
        adjusted_question_feat = self.linear1(question_feat)

        # Element wise multiply every ocr token with new question feature
        # hadamard.shape = (batch_size, max_tokens, ocr_token_dim)
        hadamard = adjusted_question_feat.unsqueeze(1) * ocr_token_feats
        assert hadamard.shape[0] == batch_size

        # Calculate the attention pre softmax weights ;
        # Calculate mask and replace 0 values with -inf 
        # pre_weights.shape = (batch_size, max_tokens) 
        pre_weights = torch.sum(hadamard, dim = 2, dtype = float)
        mask = torch.arange(ocr_token_feats.shape[1])[None, :] < num_tokens[:, None]
        pre_weights[~mask] = float('-inf')

        # Apply softmax over these
        # weights.shape = (batch_size, max_tokens) 
        weights = f.softmax(pre_weights, dim = 1)

        # Apply this attention over the ocr tokens
        # attended_feature.shape = (batch_size, 1, ocr_token_dim)
        attended_feature = torch.bmm(weights.unsqueeze(1), ocr_token_feats.type(torch.float64))

        # Final output returns shape (batch_size, ocr_token_dim)
        return attended_feature.squeeze()


if __name__ == '__main__':
    question = torch.rand(4, 768)
    ocr_features = torch.rand(4, 5, 50)
    num_tokens = torch.tensor([5, 5, 5, 5])
    ocr_text_att = VectorizedOCRTokensAndText()
    vectorized_out = ocr_text_att(question, ocr_features, num_tokens)
    loop_out = ocr_text_att.loop_forward(question, ocr_features, num_tokens)
    print(vectorized_out)
    print(loop_out[-1])