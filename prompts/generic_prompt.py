

class GENP:
    def __init__(self) -> None:

        self.example = [
            "दरिद्र",
            "दरिद्रतामा",
            "दरिद्रहरू"
        ]
        self.shot = {
    "stem": "दरिद्र",
    "definition": "poor, impoverished",
    "proper noun": "no",
    "forms": [
        "दरिद्र",
        "दरिद्रतामा",
        "दरिद्रहरू"
    ]
}

    def make_message(self, lang1, lang2, data):
        preamble = f"You are a skilled {lang1} linguist. A {lang2} linguist has come to you with what he believes is a bunch of inflected forms. Your job is to provide the stem (or stems if there are in fact multiple stems), proper noun assessment (either 'yes' or 'no'), and definition in {lang2} for each stem. For example: If the languages were {lang1} and {lang2}, and the data was\n{self.example}\nYou would return\n{self.shot}\n"
        core = f"Your turn! Please analyze this {lang1} group of words: {data}\nRemember to always respond in JSON. Always return at least one stem, one proper noun assessment, and one definition per stem, and always return each word form mapped to one stem."
        return preamble + core