from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.tokenize.treebank import TreebankWordDetokenizer

app = Flask(__name__)

def summarize_text(text, num_sentences=3):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    freq_dist = FreqDist(words)
    sentences = sent_tokenize(text)

    ranking = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq_dist:
                if i not in ranking:
                    ranking[i] = freq_dist[word]
                else:
                    ranking[i] += freq_dist[word]

    selected_sentences = sorted(ranking, key=ranking.get, reverse=True)[:num_sentences]
    summary = [sentences[i] for i in sorted(selected_sentences)]

    return TreebankWordDetokenizer().detokenize(summary)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():
    if request.method == "POST":
        inputtext = request.form["inputtext"]
        num_sentences = int(request.form.get("num_sentences", 3))
        summary = summarize_text(inputtext, num_sentences)
    return render_template("output.html", summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
