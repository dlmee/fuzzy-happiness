

class BATP:
    def __init__(self) -> None:

        self.example = ['खिड्किबाट', 'आँट', 'मुस्लो']
        self.shot = {
  "खिड्किबाट": {
    "stem": "खिड्क",
    "definition": "window",
    "proper noun": "no"
  },
  "आँट": {
    "stem": "आँट",
    "definition": "thread",
    "proper noun": "no"
  },
  "मुस्लो": {
    "stem": "मुस्ल",
    "definition": "Muslim",
    "proper noun": "yes"
  }
}


    def make_message(self, lang1, lang2, data):
        preamble = f"You are a skilled {lang1} linguist. A {lang2} linguist has come to you with what he believes is a bunch of inflected forms. Your job is to provide the stem (or stems if there are in fact multiple stems), proper noun assessment (either 'yes' or 'no'), and definition in {lang2} for each stem. For example: If the languages were {lang1} and {lang2}, and the data was\n{self.example}\nYou would return\n{self.shot}\n"
        core = f"Your turn! Please analyze this {lang1} group of words: {data}\nRemember to always respond in JSON. Always return at least one stem, one proper noun assessment, and one definition per stem, and always return each word form mapped to one stem."
        return preamble + core