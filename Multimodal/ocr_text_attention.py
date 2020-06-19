import torch, torch.nn as nn, torch.nn.functional as f

class OCRTokensAndText(nn.Module):
    def __init__(self, question_dim: int = 768, ocr_token_dim: int = 50):
        super().__init__()
        self.linear1 = nn.Linear(question_dim, ocr_token_dim)

    def forward(self, question_feat: torch.tensor, ocr_token_feats: torch.tensor) -> torch.tensor:
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
        # weights.shape = (k) 
        weights = f.softmax(torch.sum(hadamard, dim = 1))

        # Apply this attention over the ocr tokens
        # attended_feature.shape = (1, ocr_token_dim)
        attended_feature = torch.mm(weights.unsqueeze(0), ocr_token_feats)
        
        # Final output returns shape (ocr_token_dim)
        return attended_feature.squeeze()

if __name__ == '__main__':
    question = torch.rand(768)
    ocr_features = torch.rand(10, 50)
    ocr_text_att = OCRTokensAndText()
    final_attended_vector = ocr_text_att(question, ocr_features)
    breakpoint()
