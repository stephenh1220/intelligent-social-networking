
import argparse
from facial_detection.facial_detection import FacialDetection
from database.db import Database
from llm_extraction.extract_from_llm import LLMAgent
import cv2
from PIL import Image, ImageDraw
import numpy as np

def arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--phase', type=int, help='phase 1 to add to database, phase 2 to query database')
    return parser.parse_args()

background = np.ones((400, 600, 3), np.uint8) * 255  # White background
def draw_faces(frame, faces, found=False, text = 'Face Detected'):
  # Draw bounding box around each detected face
    (x, y, w, h)  = (int(x) for x in faces)

    # green if found, red if not found
    color = (0, 255, 0) if found else (0, 0, 255)
    cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), color, 2)  # Green rectangle
    
    font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a font

    # split text by \n, put text on new lines and add white background behind text. make sure the text doesnt overlap vertically
    text = text.split('\n')
    for i, t in enumerate(text):
        cv2.putText(frame, t, (int(x), int(y) + 20 + 20*i), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, t, (int(x), int(y) + 20 + 20*i), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def main(args):
    llm_agent = LLMAgent()
    response = llm_agent.generate('''Given the following conversation transcription extract the key information for each user. Respond in an easily readble format that could be displayed on a live video stream. As few words as possible for each section.
                                  \n My name is Ben. I am a student at MIT studying computer science. I am on the varsity tennis team and I like to play music, run, and hang with friends in my free time.''')

    facial_detection = FacialDetection()
    database = Database("hnsw", "cosine")

    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(1)        # Open the default camera

    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if frame is None:
            continue

        #cv2.imshow('preview', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if i%10 == 0:
            face_boxes = facial_detection.extract_face_box(frame)
            if face_boxes is None:
                continue

            detected_face = facial_detection.detect_face(frame)
            embedding = facial_detection.get_facial_embeddings(detected_face)

            if len(database.table) == 0:
                database.add_entry(embedding.numpy().reshape(-1), response)
                text = 'Face Detected'
                found = False
            else:
                value, distance = database.query_entry(embedding)
                if distance > 0.05:
                    database.add_entry(embedding.numpy().reshape(-1), response)
                    text = 'Face Detected'
                    found = False
                else:
                    text = value
                    found = True

        processed_frame = draw_faces(frame, face_boxes, found, text)

        print(len(database.table))


        # Display the resulting frame
        if i %2== 0:
            cv2.imshow('preview', processed_frame)
        
        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i+=1

    # Release capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

    


if __name__ == "__main__":
    args = arg_parser()
    main(args)