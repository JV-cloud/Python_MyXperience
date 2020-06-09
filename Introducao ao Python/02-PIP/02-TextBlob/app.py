from textblob import TextBlob

frase = TextBlob("Esse workshop está uma porcaria!")
traducao = TextBlob(str(frase.translate(to='en')))

print(traducao.sentiment)

# -POLARITY - é um valor contínuo que varia de -1.0 a 1.0, sendo -1.0 referente a 100% negativo e 1.0 a 100% positivo.
# -SUBJECTIVITY - que também é um valor contínuo que varia de 0.0 a 1.0, sendo 0.0 referente a 100% objetivo e 1.0 a 100% subjetivo. 
# Fonte: https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis
