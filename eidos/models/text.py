import torch
class TextTokenizer:
    def __init__(self,max_length=32,offset=5000):
        self.max_length=max_length
        self.offset=offset
    def encode(self,text):
        if not text:
            return []
        clean_text=text.lower().strip()
        tokens=[ord(char)+self.offset for char in clean_text if ord(char)<256]
        if len(tokens)>self.max_length:
            tokens=tokens[:self.max_length]
        return tokens
    def decode(self,tokens):
        chars=[]
        for token in tokens:
            if self.offset<=token<self.offset+256:
                chars.append(chr(token-self.offset))
        return "".join(chars)
            