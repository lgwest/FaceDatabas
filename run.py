from recognition import FaceRecognition
# pip install dlib==19.22, nej 19.22 funkar inte! använd senaste dlib

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
