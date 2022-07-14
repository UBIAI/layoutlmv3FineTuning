import os
from PIL import Image, ImageDraw, ImageFont
from .utils import image_label_2_color


def get_flattened_output(docs):
  flattened_output = []
  annotation_key = 'output'
  for doc in docs:
    flattened_output_item = {annotation_key: []}
    doc_annotation = doc[annotation_key]
    for i, span in enumerate(doc_annotation):
      if len(span['words']) > 1:
        for span_chunk in span['words']:
          flattened_output_item[annotation_key].append(
              {
                  'label': span['label'],
                  'text': span_chunk['text'],
                  'words': [span_chunk]
              }
          )
      else:
        flattened_output_item[annotation_key].append(span)
    flattened_output.append(flattened_output_item)
  return flattened_output


def annotate_image(image_path, annotation_object):
  img = None
  image = Image.open(image_path).convert('RGBA')
  tmp = image.copy()
  label2color = image_label_2_color(annotation_object)
  overlay = Image.new('RGBA', tmp.size, (0, 0, 0)+(0,))
  draw = ImageDraw.Draw(overlay)
  font = ImageFont.load_default()

  predictions = [span['label'] for span in annotation_object['output']]
  boxes = [span['words'][0]['box'] for span in annotation_object['output']]
  for prediction, box in zip(predictions, boxes):
      draw.rectangle(box, outline=label2color[prediction],
                     width=3, fill=label2color[prediction]+(int(255*0.33),))
      draw.text((box[0] + 10, box[1] - 10), text=prediction,
                fill=label2color[prediction], font=font)

  img = Image.alpha_composite(tmp, overlay)
  img = img.convert("RGB")

  image_name = os.path.basename(image_path)
  image_name = image_name[:image_name.find('.')]
  img.save(f'/content/{image_name}_inference.jpg')
