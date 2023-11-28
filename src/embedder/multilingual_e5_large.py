from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch.nn.functional as F

class MultilingualE5LargeEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

    def tokenize(self, input_text: str) -> dict[str, Tensor]:
        if len(input_text) > 512:
            print(f"Warning: input text length of {len(input_text)} exceeds max length of 512 tokens. Will be truncated to 512 tokens.")

        token_dict = self.tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return token_dict

    def embed(self, input_text: str) -> list[float]:
        def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            '''How much was each token influenced by the others + normalize by number of real tokens from attention_mask'''
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        tokens = self.tokenize(input_text)
        outputs = self.model(**tokens)
        embeddings = average_pool(outputs.last_hidden_state, tokens['attention_mask'])
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        return embeddings_norm


if __name__ == "__main__":

    # Each input text should start with "query: " or "passage: ", even for non-English texts.
    # For tasks other than retrieval, you can simply use the "query: " prefix.
    input_texts = ['query: how much protein should a female eat',
                'query: 南瓜的家常做法',
                "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
                "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"]

    embedder = MultilingualE5LargeEmbedder()

    embeddings = [embedder.embed(text) for text in input_texts]

    print(embeddings)


