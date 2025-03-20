# intelligent-social-networking
6.s079 project

## Description for Tubi
I worked on the facial recognition database portion of this project. The project leverages pre-trained facial models and LLMs into a wearable AR application designed to enhance social interaction. The database stores facial embeddings and optimizes queries using different search methods: Locality-Sensitive Hashing (LSH), Hierarchical Navigable Small World (HNSW), vector compression, and linear search. Each method supports both L2-squared distance or cosine similarity for nearest-neighbor retrieval. LSH hashes embeddings for fast lookups, HNSW builds a graph structure for efficient approximate search, vector compression reduces dimensionality for speed, and linear search iterates through stored embeddings. The code provides the architecture for use to systematically tests these methods to compare their efficiency and accuracy, which we do in the project to optimze the database’s performance.


## Reproduce main results
Run ```python main.py --phase 1``` to add to the database. This will load the phase 1 video, extract a facial embedding for each frame, average them at the end to get one vector key. It will alos take the transcript and run it through Gemini to get a condensed version which is stored as the value of the embedding key in the database.

Run ```python main.py --phase 2``` to run the inference code which extracts a facial embedding at each frame, searched the database, and visualizes the transcript key points next to the face if a match is found.


## Detecting a face
Found in `facial_detection/facial_detection_example.ipynb`
```
from facial_detection import FacialDetection
from PIL import Image
facial_detection = FacialDetection(visualize=True)
example_img = Image.open('facial_detection/example_facial_img.png').convert('RGB')
example_img.resize((400, 400))

import matplotlib.pyplot as plt
detected_face = facial_detection.detect_face(example_img)

plt.imshow(detected_face[0].permute(1, 2, 0).int().numpy())
plt.axis('off')
plt.show()


detected_face_2 = facial_detection.detect_face(example_img_2)

plt.imshow(detected_face_2[0].permute(1, 2, 0).int().numpy())
plt.axis('off')
plt.show()
```

## Extract facial embeddings
Found in `facial_detection/facial_detection_example.ipynb`
```
from facial_detection import FacialDetection
from PIL import Image
facial_detection = FacialDetection(visualize=False)
example_img = Image.open('facial_detection/example_facial_img.png').convert('RGB')
detected_face = facial_detection.detect_face(example_img)
embedding = facial_detection.get_facial_embeddings(detected_face)
```

## Use LLM to extract from transcript
Found in `llm_extraction/Gemini_LLM.ipynb`
```
import pathlib
import textwrap
import os

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

os.environ['GOOGLE_API_KEY'] = "YOUR_KEY"

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Given the following conversation transcription extract the key information for each user. \n Ben: My name is Ben. I am a student at MIT studying computer science. I am on the varsity tennis team and I like to play music, run, and hang with friends in my free time. \n Andrew: My name is Andrew. I study computer science at MIT and I play on the varsity football team. I like to hike, run, and hang with my girlfriend in my free time.")

to_markdown(response.text)
```

## Reproduce database search results
Found in `testing/test_database_creation.ipynb`