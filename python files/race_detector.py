from deepface import DeepFace

analysis = DeepFace.analyze(img_path = "/Users/alexthuruthel/sem4/Precog Task/data/img/01749.png", actions = ["gender", "race"],enforce_detection=False)

print(analysis[0]["dominant_race"])
# print(analysis2)
