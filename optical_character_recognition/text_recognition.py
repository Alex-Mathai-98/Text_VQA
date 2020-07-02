import torch, torch.nn as nn
from optical_character_recognition import utils, models
from torch.autograd import Variable
from PIL import Image

class TextRecognition(nn.Module):
    def __init__(self, model_path = './optical_character_recognition/pretrained_models/crnn.pth', alphabet = '0123456789abcdefghijklmnopqrstuvwxyz. ', model = 'CRNN'):
        super().__init__()
        if model == 'CRNN':
            self.model = models.CRNN(32, 1, 37, 256)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.converter = utils.strLabelConverter(alphabet)
        self.transformer = utils.resizeNormalize((100, 32))
        self.model.eval()

    def forward(self, image):
        image = Image.fromarray(image)
        image = image.convert('L')
        image = self.transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        predictions = self.model(image)
        _, predictions = predictions.max(2)
        predictions = predictions.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([predictions.size(0)]))
        raw_pred = self.converter.decode(predictions.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(predictions.data, preds_size.data, raw=False)
        return raw_pred, sim_pred

if __name__ == '__main__':
    img_path = './Data/demos/demo_2.jpg'
    text_recognizer = TextRecognition()
    raw_pred, sim_pred = text_recognizer(img_path)
    breakpoint()