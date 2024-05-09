

class ALIP:
    def __init__(self) -> None:

        self.example = {
            "eng 53": "When they had crossed over, they landed at Gennesaret and anchored there.",
            "nep 53-54": "अनि तिनीहरूले झीलको पारी पट्टी पार (नाघेर) गरेर गनेसरतमा पुगेर डुङ्गालाई बाँधे । येशू डुङ्गाबाट निस्‍किने बित्तिकै त्‍यहाँका मानिसहरू सबैले उहाँलाई चिने ।",
            "eng 54": "As soon as they got out of the boat, people recognized Jesus."
    }
        self.shot = {
        "eng 53": "When they had crossed over, they landed at Gennesaret and anchored there.",
        "eng 54": "As soon as they got out of the boat, people recognized Jesus.",
        "nep 53": "अनि तिनीहरूले झीलको पारी पट्टी पार (नाघेर) गरेर गनेसरतमा पुगेर डुङ्गालाई बाँधे ।",
        "nep 54": "येशू डुङ्गाबाट निस्‍किने बित्तिकै त्‍यहाँका मानिसहरू सबैले उहाँलाई चिने ।"
} 

    def make_message(self, lang1, lang2, data):
        preamble = f"You are a skilled {lang1} linguist. You have some misaligned data, and your job is to realign it. Fortunately you have the original alginment from {lang2}. For example: If the languages were {lang1} and {lang2}, and the data was\n{self.example}\nYou would return\n{self.shot}\n"
        core = f"Your turn! Please realign the following data {data}. Always respond in JSON."
        return preamble + core