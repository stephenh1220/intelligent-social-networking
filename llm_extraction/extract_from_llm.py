import google.generativeai as genai
import json
from IPython.display import Markdown
import textwrap

class LLMAgent:

    def __init__(self):
        with open("llm_extraction/secret.json", 'r') as f:
            json_data = json.load(f)

        GOOGLE_API_KEY = json_data['GOOGLE_API_KEY']
        genai.configure(api_key=GOOGLE_API_KEY)

        self.model = genai.GenerativeModel('gemini-pro')

    def to_markdown(self, text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
    

    def generate(self, prompt):
        response = self.model.generate_content(prompt)
        #return self.to_markdown(response.text)
        return response.text
