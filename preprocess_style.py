import re
import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj\style\filtered"
LINE_MIN_LEN = 20

acronyms = {
    "אע\"פ": "אף על פי",
    "שנא'": "שנאמר",
    "ב\"ש": "בית שמאי",
    "ב\"ה": "בית הלל",
    "א\"ר": "אמר רבי",
    "ר'": "רבי",
    "עקיבה": "עקיבא",
    "רשב\"ג": "רבן שמעון בן גמליאל",
    "חכ\"א": "חכמים אומרים",
    "ר\"ש": "רבי שמעון",
    "ר\"א": "רבי אליעזר",
    "ב\"ר": "בן רבי",
    "אח\"כ": "אחר כך",
    "בד\"א": "במה דברים אמורים",
    "ב\"ד": "בית דין",
    "אפי'": "אפילו",
    "א'": "אחד",
    "ע\"ג": "על גבי",
    "ר\"ע": "רבי עקיבא",
    "יו\"ט": "יום טוב",
    "ת\"ל": "תלמוד לומר",
    "ע\"מ": "על מנת",
    "אא\"כ": "אלא אם כן",
    "רה\"ר": "רשות הרבים",
    "רה\"י": "רשות היחיד",
    "רשב\"א": "רבי שמעון בן אלעזר",
    "ע\"י": "על ידי",
    "בעה\"ב": "בעל הבית",
    "ע\"פ": "על פי",
    "א\"צ": "אין צריך",
    "ב\"א": "בן אלעזר",
    "ר\"ג": "רבן גמליאל",
    "ריב\"ז": "רבן יוחנן בן זכאי",
    "א\"כ": "אם כן",
    "ר\"י": "רבי יהודה",
    "אעפ\"כ": "אף על פי כן",
    "ע\"ה": "עם הארץ",
    "רש\"א": "רבי שמעון אומר",
    "ק\"ו": "קל וחומר",
    "יוה\"כ": "יום הכיפורים",
    "י\"ט": "יום טוב",
    "ע\"ש": "ערב שבת",
    "נ\"ש": "נזק שלם",
    "ח\"נ": "חצי נזק",
    "ה\"ז": "הרי זה",
    "ד\"א": "דבר אחר",
    "יהושוע": "יהושע",
    "ר\"מ": "רבי מאיר",
    "חביר": "חבר",
    "אפלו": "אפילו",
    "חיב": "חייב",
    "מניין": "מנין",
    "שנ'": "שנאמר",
    "ישר'": "ישראל",
    "המקו'": "המקום",
    "וגו'": "וגומר",
    "וכו'": "וכולי",
    "ת\"ר": "תנו רבנן",
    "אמ'": "אמר",
    "או'": "אומר",
    "דתכ'": "דכתיב",
    "ע\"ז": "עבודה זרה",
    "אעפ\"י": "אף על פי",
    "א\"ל": "אמר ליה",
    "הקב\"ה": "הקדוש ברוך הוא",
    "רבש\"ע": "רבונו של עולם",
    "הה\"ד": "הדא הוא דכתיב",
    "קמ\"ל": "קא משמע לן",
    "מ\"ט": "מאי טעמא",
    "ת\"ש": "תא שמע",
    "ש\"מ": "שמע מינה",
    "ה\"נ": "הכא נמי",
    "ס\"ד": "סלקא דעתך",
    "זש\"ה": "זה שאמר הכתוב",
    "מ\"ד": "מאן דאמר",
    "ר\"ל": "ריש לקיש",
    "ה\"מ": "הני מילי",
    "ה\"ק": "הכי קאמר",
    "כ\"ש": "כל שכן",
    "דכ\"ע": "דכולי עלמא",
    "ה\"א": "הוה אמינא",
    "ריב\"ל": "רבי יהושע בן לוי",
    "מ\"מ": "מכל מקום",
    "ה\"ד": "היכי דמי",
    "ר\"ח": "ראש חודש",
    "ת\"ח": "תלמידי חכמים",
    "ת\"ק": "תנא קמא",
    "ק\"ש": "קריאת שמע",
    "כמ\"ד": "כמאן דאמר",
    "א\"י": "ארץ ישראל",
    "י\"א": "יש אומרים",
    "כ\"ת": "כי תימא",
    "ל\"ש": "לא שנא",
    "ל\"ק": "לא קשיא",
    "א\"נ": "אי נמי",
    "אצ\"ל": "אין צריך לומר",
    "בש\"א": "בית שמאי אומרים",
    "בה\"א": "בית הלל אומרים",
    "רשב\"י": "רבי שמעון בר יוחאי",
    "עוה\"ב": "עולם הבא",
    "עאכ\"ו": "על אחת כמה וכמה",
    "מנה\"מ": "מנא הני מילי",
    "בהמ\"ק": "בית המקדש",
    "אפ\"ה": "אפילו הכי",
    "יצה\"ר": "יצר הרע",
    "כמ\"ש": "כמה שכתוב",
    "אלוקים": "אלהים"



}



def filter_line(line):
    if len(line) < LINE_MIN_LEN:
        return False
    if len(line.split()) < 5:
        return False
    if not re.search("[א-תםןץףך]", line):
        return False
    if re.search("[a-zA-Z]", line):
        return False
    return True

def clean(text):
    result = re.sub(": ", ". ", text)
    result = re.sub(" ?\[.*?\]", "", result)
    result = re.sub(" ?<.*?>", "", result)
    result = re.sub(" ?\(.*?\)", "", result)
    result = re.sub("[^א-ת\.\,\'\"\s]", "", result) #no nikkud
    for k in sorted(acronyms, key=len, reverse=True):
        result = result.replace(k, acronyms[k])
    return result

if __name__ == "__main__":
    paths = glob.glob(os.path.join(BASE_PATH, r"*\*\Hebrew\merged.txt"))
    paths += glob.glob(os.path.join(BASE_PATH, r"*\*\*\Hebrew\merged.txt"))
    paths += glob.glob(os.path.join(BASE_PATH, r"*\*\*\*\Hebrew\merged.txt"))
    paths += glob.glob(os.path.join(BASE_PATH, r"*\*\*\*\*\Hebrew\merged.txt"))
    paths = sorted(paths)
    print(len(paths))

CHUNK_SIZE = 50

all_chunks = []
all_labels = []
all_books = []
all_chunk_ids = []

def preprocess_single_file(path, do_original_paragraphs=False, do_limit_chunk_size=False):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()#[:10000]

    data = clean(data)
    lines = data.split("\n")
    lines = list(filter(filter_line, lines))
    data = "\n".join(lines)
    # data = clean(data)

    data = re.sub("שנאמר [א-תםןףךץ]+ [א-תםןףךץ]+,", "שנאמר", data)


    if do_original_paragraphs:
        chunks = data.split("\n")
        if do_limit_chunk_size:
            results = []
            for chunk in chunks:
                if len(chunk.split()) > CHUNK_SIZE:
                    data = chunk.split()
                    results += [' '.join(data[i:i + CHUNK_SIZE]) for i in range(0, len(data), CHUNK_SIZE)]
                    if len(results[-1].split()) < CHUNK_SIZE // 2:
                        results = results[:-2] + [" ".join([results[-2], results[-1]])]
                else:
                    results += [chunk]

            chunks = results
    else:
        data = data.split()
        chunks = [' '.join(data[i:i + CHUNK_SIZE]) for i in range(0, len(data), CHUNK_SIZE)]
    return chunks


def preprocess_single_file_raw(path, delim="\n"):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()#[:10000]

    data = clean(data)
    lines = data.split("\n")
    lines = list(filter(filter_line, lines))
    data = delim.join(lines)
    # data = clean(data)

    data = re.sub("שנאמר [א-תםןףךץ]+ [א-תםןףךץ]+,", "שנאמר", data)

    return data

# def create_dataset_from_paths(paths, outpath=r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset_220410.json"):
def create_dataset_from_paths(paths, outpath=r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset_220424.json"):
    global all_chunks, all_labels, all_books, all_chunk_ids

    for path in tqdm(paths):
        label = int(path[len(BASE_PATH):].split("\\")[1])
        book = path[len(BASE_PATH):].split("\\")[2]
        base = os.path.basename(path)
        tractate = base[base.find("_")+1:].split(".")[0]

        chunks = preprocess_single_file(path)

        all_chunks += chunks
        all_labels += [label] * len(chunks)
        all_books += [book] * len(chunks)
        all_chunk_ids += range(len(chunks))

    df = pd.DataFrame(data=np.array([all_chunks, all_labels, all_books, all_chunk_ids]).T, columns=["text", "label", "book", "chunk_id"])
    if outpath:
        df.to_json(outpath)
    return df
    # df.to_json(os.path.join(BASE_PATH, r"mishna\dataset.json"))

if __name__ == "__main__":
    create_dataset_from_paths(paths)
