import torch, torch.nn as nn
import optical_character_recognition.utils as utils
import os, random, argparse, cv2, numpy as np
from optical_character_recognition.text_detection import TextDetection
from optical_character_recognition.text_recognition import TextRecognition

class GloveEmbeddings(object):
    def __init__(self, glove_file = 'Data/glove.6B.50d.txt'):
        """
        GloVe embedder class to load the file with the mappings  
        """
        with open(glove_file, 'r') as f:
            self.words = set()
            self.word_to_vec_map = {} 
            n = 0           
            unk_token = 0

            for line in f:
                n += 1 
                line = line.strip().split()
                curr_word = line[0]
                self.words.add(curr_word)
                self.word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
                unk_token += np.array(line[1:], dtype=np.float64)
            
            self.unk_token = unk_token / n
        
        assert self.unk_token.shape == np.array(line[1:]).shape

    def get_embedding(self, word):
        """
        Get the embedding for the given word using the initialized mapping 
        """
        word = str(word).lower()
        try: embedding = self.word_to_vec_map[word]
        except KeyError: embedding = self.unk_token

        return embedding

class EndToEndOCR(nn.Module):
    def __init__(self):
        """
        Initialize Text Detection and Recognition for the complete OCR model 
        """
        super().__init__()  
        self.text_detector = TextDetection()
        self.text_recognizer = TextRecognition()
        self.embedder = GloveEmbeddings() # Use glove_file = FILEPATH to be be able to use
                                          # that file for embeddings (6B.50d default)
        if torch.cuda.is_available():
            self.text_detector.cuda(), self.text_recognizer.cuda()
    
    def forward(self, image_path: str):
        """
        Takes an input image_path and gives out a list of all tokens in that image along with
        a list of all the embeddings of these tokens. 
        
        Model uses the end to end pipeline with no pre or post processing additional to that 
        done by the model themselves  
        """
        ocr_tokens = list()
        image = cv2.imread(image_path)
        boxes, _, _, _ = self.text_detector(image_path, refine = False)
        boxes = boxes.astype(np.int32)

        for box in boxes:
            x_max, x_min = max(box[:, 0]), min(box[:, 0])
            y_max, y_min = max(box[:, 1]), min(box[:, 1])
            tmp = image[y_min:y_max, x_min:x_max]
            _, sim_pred = self.text_recognizer(tmp)
            ocr_tokens.append(sim_pred)

        embeddings = []
        for token in ocr_tokens:
            embeddings.append(self.embedder.get_embedding(token))

        return ocr_tokens, embeddings, len(ocr_tokens)
    
    def visualize_predictions(self, image_path):
        """
        Visualize predictions as the CTC reconstruction instead of just outputting the token ; 
        and also display text region crops one by one
        """
        image = cv2.imread(image_path)
        boxes, _, _, _ = self.text_detector(image_path)
        boxes = boxes.astype(np.int32)

        for box in boxes:
            x_max, x_min = max(box[:, 0]), min(box[:, 0])
            y_max, y_min = max(box[:, 1]), min(box[:, 1])
            cv2.polylines(image, [box], True, (0, 0, 255))
            tmp = image[y_min:y_max, x_min:x_max]
            raw_pred, sim_pred = self.text_recognizer(tmp)

            print("Predicted raw word ======> {} ======> {}".format(raw_pred, sim_pred))
            cv2.imshow("Token", tmp)
            cv2.waitKey()

        cv2.imshow("Complete Image", image)
        cv2.waitKey()

    def get_heirarchal_preds(self, image_path):
        """
        Given an image link, generate refined text boxes (boxes with phrases) first. Each of these phrases 
        maps to its constituent word boxes. If the constituent word is written top to bottom, then that word 
        also maps to each of its constituent letter boxes. 

        This creates a 3 levelled heirarchy as " phrases -> words -> letters ", stored as a list of lists of lists. 
        Predictions are then found at the lowest heirarchy levels (either at letter level or word level depending 
        on orientation), and are aggregated as per the order in the phrase box. The ordering assumes that all ocr 
        tokens are read either from left to right or from top to bottom (reasonable for english). 

        To avoid comprehensive explanation, each function called from utils has also been documented in the 
        utils file. This has been heavily documented because extensive pre and post processing was done to the 
        previous existing model.
        """

        # Get the refined text boxes and region scores 
        boxes, polys, region_score, affinity_score = self.text_detector(image_path)
        image = utils.loadImage(image_path)

        # Pre-processing for straightening text at an angle for left to right readability 
        refined_box_cuts, scores = utils.get_straightened_boxes(image, region_score, boxes)

        # Break phrases into words 
        cleaned_refined_box_cuts, refined_words, refined_word_heatmaps, orientations, final_coords, overall_boxes = utils.word_level_breakdown(refined_box_cuts, self.text_detector, boxes)
        tokens, locations, ind1 = [], [], 0

        # Iterate through all phrases
        for word, maps, rot in zip(refined_words, refined_word_heatmaps, orientations):
            string, ind2 = "", 0

            # Iterate through constituent words for the phrases
            for w, m in zip(word, maps):

                # Check if the word is read top to bottom (rot = False)
                if not rot:

                    # Get character level boxes and predictions 
                    letter_boxes, _, vert_pred = utils.get_characters_top_down(w, m, self.text_recognizer)

                    # Aggregate together to make word from constituent letters
                    if len(vert_pred) > 1:
                        tokens.append(vert_pred)
                        locations.append(final_coords[ind1][ind2])
                
                # If left-to-right, generally apply text recognition on word
                _, out = self.text_recognizer(w)
                tokens.append(out)
                locations.append(final_coords[ind1][ind2])
                ind2 += 1
                string += " " + out

            # Apart from the word level tokens, also append the whole phrase tokens, 
            # This might contain things like 'thank you' or 'good morning' 
            tokens.append(string[1:])
            locations.append(overall_boxes[ind1])
            ind1 += 1

        # Get rid of repeated tokens and sort by decreasing token length 
        cleaned_tokens = sorted(list(set(tokens)), key = lambda x: -len(x))
        return tokens, cleaned_tokens, refined_words, locations

if __name__ == '__main__':

    # Parse single argument for custom image path
    parser = argparse.ArgumentParser(description='Complete end-to-end OCR inference')
    parser.add_argument('--image_path', default='Data/demos/', type=str, help='folder path to input images')
    args = parser.parse_args()

    # If directory, get random image in that directory otherwise just use the image path given
    if os.path.isdir(args.image_path): image_path = os.path.join(args.image_path, random.choice(os.listdir(args.image_path)))
    else: image_path = args.image_path

    # Instantiate OCR model, get tokens in the image (acc. to the model), visualize what the model saw as what
    ocr = EndToEndOCR()
    tokens, clean, refi, locs = ocr.get_heirarchal_preds(image_path)
    breakpoint()