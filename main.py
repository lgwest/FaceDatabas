import cv2
import face_recognition
import numpy as np
import os
from face_recognition import face_locations, face_encodings

# Skapa en ny databas om den inte redan finns någon
if not os.path.exists("database.pkl"):
    database = {}
else:
    with open("database.pkl", "rb") as f:
        database = load(f)

# Starta kameran
cap = cv2.VideoCapture(0)

# Kör i en loop tills användaren trycker på q
while True:
    # Ta ett kort från kameran
    ret, frame = cap.read()

    # Konvertera bilden till gråton
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Hitta ansikten i bilden
    faces = face_recognition.face_locations(grayscale_image, 1)

    # För varje ansikte
    for (top, right, bottom, left) in faces:
        # Rita en rektangel runt ansiktet
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Hitta ansiktets kod
        encoding = face_recognition.face_encodings(frame, [locations[0]])[0]

        # Jämför koden med koderna i databasen
        matches = np.array([np.linalg.norm(encoding - face) for face in database])

        # Hitta det bästa matchet
        best_match = np.argmin(matches)

        # Om det finns ett bra match
        if matches[best_match] < 0.6:
            # Skriv ut namnet på personen
            print(database[best_match][1])

        else:
            # Ansiktet är inte i databasen, lägg till det
            new_face = (encoding, input("Skriv in namnet på personen: "))
            database[new_face[1]] = new_face[0]
            with open("database.pkl", "wb") as f:
                dump(database, f)

    # Skriv ut antalet ansikten som identifierades
    cv2.putText(frame, f"Antal ansikten: {len(faces)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Visa bilden
    cv2.imshow("Face Recognition", frame)

    # Vänta på tangenttryckning
    key = cv2.waitKey(1)

    # Om användaren trycker på q, avsluta
    if key == ord("q"):
        break

# Stäng kameran
cap.release()

# Stäng alla fönster
cv2.destroyAllWindows()
