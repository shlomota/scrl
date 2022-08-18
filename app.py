from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, BertTokenizer
from BEREL.rabtokenizer import RabbinicTokenizer
import streamlit as st
from transformers_interpret import SequenceClassificationExplainer
import torch
import pandas as pd
import re

def clean_text(text, do_re=True):
  if do_re:
    text = re.sub("\w+,\w+", "", text)
  text = text.replace(" \"", " ").replace("\" ", " ")
  # text = text.replace(" \'", " ").replace("\' ", " ")
  result = text.replace(",", "")
  result = result.replace(".", "")
  result = result.replace("?", "")
  result = result.replace("!", "")
  return result
  
  
def clean_html(result):
        result = result.replace("[CLS]", "").replace("[SEP]", "")
        result = result.replace("100%", "1000px")
        return result


model_name = r"./BEREL"
# model_name = r"/content/drive/MyDrive/tanna/BEREL"
tokenizer = RabbinicTokenizer(BertTokenizer.from_pretrained(model_name))
id2label = {0: 'Mishnah', 1: 'Midrash_Halakha', 2: 'Jerusalem_Talmud', 3: 'Babylonian_Talmud', 4: 'Midrash_Aggadah', 5: 'Midrash_Tanchuma'}
label2id = {v:k for k,v in id2label.items()}
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6, id2label=id2label, label2id=label2id)


# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# With both the model and tokenizer initialized we are now able to get explanations on an example text.

default_text = "אלו דברים שאין להם שיעור הפאה והביכורים והראיון וגמילות חסדים ותלמוד תורה"

st.title("SCRL: Style Classification for Rabbinic Literature")

st.markdown("""
<style>
textarea {
  unicode-bidi:bidi-override;
  direction: RTL;
}
</style>
    """, unsafe_allow_html=True)

#st.markdown("""
#<style>
#p {
#  unicode-bidi:bidi-override;
#  direction: RTL;
#  font-family: 'David Libre';
#}
#</style>
#    """, unsafe_allow_html=True)

st.markdown("""
        <style>
        table {
              width: 150%;
              }
        </style>
            """, unsafe_allow_html=True)

#label = "Enter a verse here (no punctuation or vowels):"
label = "Enter a passage from Rabbinic literature:"
text = st.text_area(label, value=default_text)

text = clean_text(text)




#do_all = True
#do_all = st.checkbox("All classes", value=False, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)
no_explain = st.checkbox("No explanation", value=False)
do_all = st.checkbox("All classes", value=False)

if no_explain:
    X_train_tokenized = tokenizer([text], padding=True, truncation=True, max_length=512)
    probs = model(torch.tensor(X_train_tokenized["input_ids"])).logits.softmax(dim=-1).squeeze().tolist()

    df = pd.DataFrame(columns=["label", "probability"])
    df.label = [id2label[i] for i in range(len(id2label))]
    df.probability = probs
    st.table(data=df)



elif do_all:
    for class_name in list(model.config.id2label.values()):
        cls_explainer = SequenceClassificationExplainer(
            model,
            tokenizer)
        word_attributions = cls_explainer(text, class_name=class_name)
        res = cls_explainer.visualize()._repr_html_()
        res = clean_html(res)
        st.markdown(res, unsafe_allow_html=True)

else:
    cls_explainer = SequenceClassificationExplainer(
        model,
        tokenizer)
    word_attributions = cls_explainer(text)
    res = cls_explainer.visualize()._repr_html_()
    res = clean_html(res)
    st.markdown(res, unsafe_allow_html=True)

# print(cls_explainer.predicted_class_index)
# print(cls_explainer.predicted_class_name)
