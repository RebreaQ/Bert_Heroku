import os
from flask import Flask, render_template
from flask import request
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

tokenizerchange = AutoTokenizer.from_pretrained("Rebreak/bert_news_class")

modelchange = AutoModelForSequenceClassification.from_pretrained("Rebreak/bert_news_class")

#tokenizersent = AutoTokenizer.from_pretrained("ProsusAI/finbert")

#modelsent = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


def predict(text, model, tokenizer):
    pipeline = TextClassificationPipeline(model=model,
                                          tokenizer=tokenizer,
                                          framework='pt',
                                          )
    return "Don't change" if pipeline(text)[0]['score'] > 0.90 else "Change"


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        #result = []
        form = request.form
        bert_abstract = form['text']
        result = predict(bert_abstract, modelchange, tokenizerchange)
        #result.append(predict(bert_abstract, modelsent, tokenizersent))

        return render_template("index.html", result=result)

    # return answer
    return render_template("index.html")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
