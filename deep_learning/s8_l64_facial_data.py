import numpy as np

# 0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, 6 = Neutral
def get_emotion(emotion_id):
   # emotion_id = np.round(emotion_id)
   match (str(int(emotion_id))):
       case "0":
           return "Angry"
       case "1":
           return "Disgust"
       case "2":
           return "Fear"
       case "3":
           return "Happy"
       case "4":
           return "Sad"
       case "5":
           return "Surprise"
       case "6":
           return "Neutral"
       case _:
           return "Not supported"  # 0 is the default case if x is not found

def get_emotion_if(emotion_id):
    if emotion_id==0:
           return "Angry"
    elif emotion_id==1:
           return "Disgust"
    elif emotion_id == 2:
        return "Fear"
    elif emotion_id == 3:
        return "Happy"
    elif emotion_id == 4:
        return "Sad"
    elif emotion_id == 5:
        return "Surprise"
    elif emotion_id == 6:
        return "Neutral"
    else:
        return "Not supported"

def get_data()  ->  tuple[np.ndarray,  np.ndarray]:
    Y = []
    X = []
    first = True
    i = 0
    # downloaded from
    for line in open('../large_files/fer2013.csv'):
        i += 1
        if first:  # column names
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                if i % 1000 == 0:
                    print("i:", i)
                Y.append(y)
                try:
                    X.append([int(p) for p in row[1].split()])
                except Exception:
                    print("row[1]:", row[1])
    return np.array(X) / 255.0, np.array(Y)


