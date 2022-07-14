import os
import pandas as pd

def run_tesseract_on_image(image_path):  # -> tsv output path
  image_name = os.path.basename(image_path)
  image_name = image_name[:image_name.find('.')]
  error_code = os.system(f'''
  tesseract "{image_path}" "/content/{image_name}" -l eng tsv
  ''')
  if not error_code:
    return f"/content/{image_name}.tsv"
  else:
    raise ValueError('Tesseract OCR Error please verify image format PNG,JPG,JPEG')


def clean_tesseract_output(tsv_output_path):
  ocr_df = pd.read_csv(tsv_output_path, sep='\t')
  ocr_df = ocr_df.dropna()
  ocr_df = ocr_df.drop(ocr_df[ocr_df.text.str.strip() == ''].index)
  text_output = ' '.join(ocr_df.text.tolist())
  words = []
  for index, row in ocr_df.iterrows():
    word = {}
    origin_box = [row['left'], row['top'], row['left'] +
                  row['width'], row['top']+row['height']]
    word['word_text'] = row['text']
    word['word_box'] = origin_box
    words.append(word)
  return words


def prepare_batch_for_inference(image_paths):
  # tesseract_outputs is a list of paths
  inference_batch = dict()
  tesseract_outputs = [run_tesseract_on_image(
      image_path) for image_path in image_paths]
  # clean_outputs is a list of lists
  clean_outputs = [clean_tesseract_output(
      tsv_path) for tsv_path in tesseract_outputs]
  word_lists = [[word['word_text'] for word in clean_output]
                for clean_output in clean_outputs]
  boxes_lists = [[word['word_box'] for word in clean_output]
                 for clean_output in clean_outputs]
  inference_batch = {
      "image_path": image_paths,
      "bboxes": boxes_lists,
      "words": word_lists
  }
  return inference_batch
