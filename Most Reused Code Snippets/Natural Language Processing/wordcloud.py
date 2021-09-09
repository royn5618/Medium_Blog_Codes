from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(list_of_tokens):
  wc = WordCloud(background_color="black", 
                 max_words=100, 
                 width=1000, 
                 height=600, 
                 random_state=1).generate(list_of_tokens)

  plt.figure(figsize=(15,15))
  plt.imshow(wc)
  plt.axis("off")
  plt.title("Title of Wordcloud")
  plt.show()
