import cv2
import os
import uuid
import face_recognition
import pyttsx3
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from google.cloud import dialogflow_v2 as dialogflow  # Import Dialogflow library

# Constants
DB_PATH = 'sqlite:///faces.db'
FACES_DIR = './faces'
MODEL_PATH = "model.xml"
ATTENDANCE_LOG_FILE = "attendance_log.txt"
FEMALE_VOICE_ID = "female"
DIALOGFLOW_PROJECT_ID = "your-project-id"  # Your Dialogflow project ID

# Set up SQLAlchemy ORM
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    image_path = Column(String)

class Attendance(Base):
    __tablename__ = 'attendance'
    
    id = Column(Integer, primary_key=True)
    user_name = Column(String)
    date = Column(String)
    time = Column(String)

    def __init__(self, user_name):
        self.user_name = user_name
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.time = datetime.now().strftime("%H:%M:%S")

# Create the database engine and session
engine = create_engine(DB_PATH)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Class to handle the entire face recognition system
class FaceRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(MODEL_PATH)
        self.session = Session()
        self.known_face_encodings = []
        self.known_face_names = []
        self.processed_names = set()
        self.tts_engine = pyttsx3.init()
        self._set_tts_properties()
        self.load_known_faces()

    def _set_tts_properties(self):
        """Set up voice properties (female voice, slower speed)."""
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if FEMALE_VOICE_ID in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
        self.tts_engine.setProperty('rate', 150)

    def load_known_faces(self):
        """Load all registered faces from the database."""
        users = self.session.query(User).all()
        self.known_face_encodings.clear()
        self.known_face_names.clear()

        for user in users:
            if os.path.exists(user.image_path):
                face_image = face_recognition.load_image_file(user.image_path)
                face_encoding = face_recognition.face_encodings(face_image)
                if face_encoding:  # Check if face encoding was successful
                    self.known_face_encodings.append(face_encoding[0])
                    self.known_face_names.append(user.name)
                else:
                    print(f"Warning: No face encoding found for {user.name}.")
            else:
                print(f"Warning: File not found for {user.name} at {user.image_path}")

    def register_new_face(self):
        """Register a new face and save it to the database."""
        cam = cv2.VideoCapture(0)
        print("Press 'q' to exit registration mode.")

        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to capture image")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi = img[y:y + h, x:x + w]
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_roi)

                if face_encodings:
                    face_encoding = face_encodings[0]
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    
                    if any(matches):
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]
                        cv2.putText(img, f"Already Registered: {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    else:
                        cv2.putText(img, "New face detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.imshow("Face Registration", img)
                        print("New face detected. Please enter your name:")
                        name = input("Enter your name: ").strip()
                        
                        if name:  # Ensure the name is not empty
                            unique_id = str(uuid.uuid4())
                            image_path = os.path.join(FACES_DIR, f"{unique_id}.jpg")
                            cv2.imwrite(image_path, roi)
                            new_user = User(name=name, image_path=image_path)
                            self.session.add(new_user)
                            self.session.commit()
                            print(f"Face for {name} registered successfully.")
                            self.known_face_encodings.append(face_encoding)
                            self.known_face_names.append(name)

            cv2.imshow("Face Registration", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

    def recognize_faces(self):
        """Recognize faces and greet known users."""
        self.load_known_faces()
        cam = cv2.VideoCapture(0)
        print("Press 'q' to exit recognition mode.")

        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to capture image")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi = img[y:y + h, x:x + w]
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_roi)

                if face_encodings:
                    face_encoding = face_encodings[0]
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    if any(matches):
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]

                        if name not in self.processed_names:
                            welcome_message = f"Welcome {name}!"
                            self.tts_engine.say(welcome_message)
                            self.tts_engine.runAndWait()
                            self.processed_names.add(name)
                            self.log_attendance(name)

                    cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Face Recognition", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

    def log_attendance(self, name):
        """Log the attendance to the database."""
        attendance_record = Attendance(user_name=name)
        self.session.add(attendance_record)
        self.session.commit()
        print(f"Attendance logged for {name}.")

    def chat_with_dialogflow(self, text):
        """Interact with Dialogflow API."""
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(DIALOGFLOW_PROJECT_ID, uuid.uuid4().hex)

        text_input = dialogflow.TextInput(text=text, language_code='en')
        query_input = dialogflow.QueryInput(text=text_input)

        response = session_client.detect_intent(session=session, query_input=query_input)
        return response.query_result.fulfillment_text

    def run(self):
        """Main function to run the system."""
        while True:
            print("Choose an option:")
            print("1. Register a new face")
            print("2. Recognize faces")
            print("3. Chat with Dialogflow")  # Add this option
            print("q. Quit")
            choice = input("Enter your choice: ")

            if choice == "1":
                self.register_new_face()
            elif choice == "2":
                self.recognize_faces()
            elif choice == "3":  # Dialogflow interaction
                user_input = input("Enter your message: ")
                response = self.chat_with_dialogflow(user_input)
                print(f"Dialogflow Response: {response}")
            elif choice.lower() == "q":
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    face_recognition_system = FaceRecognitionSystem()
    face_recognition_system.run()
