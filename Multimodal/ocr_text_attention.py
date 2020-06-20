import torch, torch.nn as nn, torch.nn.functional as f
import random, numpy as np

class VectorizedOCRTokensAndText(nn.Module):
    def __init__(self, question_dim: int = 768, ocr_token_dim: int = 50):
        super().__init__()
        self.linear1 = nn.Linear(question_dim, ocr_token_dim)

    def loop_forward(self, question_feat: torch.tensor, ocr_token_feats: torch.tensor) -> torch.tensor:
        """
        question_feat:    question feature of size - question_dim
        ocr_token_feats:  tensor of all ocr token embeddings of size - (k, ocr_token_dim)
        """
 
        # Question feature needs to match ocr token embedding size
        # Change size with linear layer
        adjusted_question_feat = self.linear1(question_feat)

        # Element wise multiply every ocr token with new question feature
        # hadamard.shape = (k, ocr_token_dim)
        hadamard = adjusted_question_feat.unsqueeze(0) * ocr_token_feats

        # Calculate the attention weights ; 
        # pre_weights.shape = (k) 
        pre_weights = torch.sum(hadamard, dim = 1, dtype = float)

        # Apply softmax
        weights = f.softmax(pre_weights, dim = 0)

        # Apply this attention over the ocr tokens
        # attended_feature.shape = (1, ocr_token_dim)
        attended_feature = torch.mm(weights.unsqueeze(0), ocr_token_feats.type(torch.float64))
        
        return attended_feature.squeeze()

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

    # Alex test for obj ------------------------------------
    MAX_OBJECTS = 5
    batch_size = 3
    IMG_DIM = 2048
    QUEST_DIM = 768

    text_inp = torch.randn((3,QUEST_DIM))
    img_inp = [ torch.randn((2,IMG_DIM)), torch.randn((4,IMG_DIM)), torch.randn((3,IMG_DIM))] 
    num_objects = torch.tensor([2,4,3])

    new_img_inp = torch.zeros( (batch_size,MAX_OBJECTS,IMG_DIM) )
    new_img_inp[0,:2,:] = img_inp[0]
    new_img_inp[1,:4,:] = img_inp[1]
    new_img_inp[2,:3,:] = img_inp[2]

    ocr_text_att = VectorizedOCRTokensAndText(question_dim= QUEST_DIM, ocr_token_dim= IMG_DIM)
    vectorized_out = ocr_text_att(text_inp, new_img_inp, num_objects)

    loop_out = []
    for n, i in enumerate(img_inp):
        loop_out.append(ocr_text_att.loop_forward(text_inp[n], i))
    loop_out = torch.stack(loop_out)

    breakpoint()