import os
import cv2
import time
import spacy
import textacy.extract
from pathlib import Path
import numpy as np
from datetime import datetime

class ISLTranslator:
    def __init__(self, video_directory="video", output_directory="output"):
        self.sign_word_videos = {
            "0": "0.mp4",
            "1": "1.mp4",
            "2": "2.mp4",
            "3": "3.mp4",
            "4": "4.mp4",
            "5": "5.mp4",
            "6": "6.mp4",
            "7": "7.mp4",
            "8": "8.mp4",
            "9": "9.mp4",
            "a": "A.mp4",
            "after": "After.mp4",
            "again": "Again.mp4",
            "against": "Against.mp4",
            "age": "Age.mp4",
            "all": "All.mp4",
            "alone": "Alone.mp4",
            "also": "Also.mp4",
            "and": "And.mp4",
            "ask": "Ask.mp4",
            "at": "At.mp4",
            "b": "B.mp4",
            "be": "Be.mp4",
            "beautiful": "Beautiful.mp4",
            "before": "Before.mp4",
            "best": "Best.mp4",
            "better": "Better.mp4",
            "busy": "Busy.mp4",
            "but": "But.mp4",
            "bye": "Bye.mp4",
            "c": "C.mp4",
            "can": "Can.mp4",
            "cannot": "Cannot.mp4",
            "change": "Change.mp4",
            "college": "College.mp4",
            "come": "Come.mp4",
            "computer": "Computer.mp4",
            "d": "D.mp4",
            "day": "Day.mp4",
            "distance": "Distance.mp4",
            "do not": "Do Not.mp4",
            "do": "Do.mp4",
            "does not": "Does Not.mp4",
            "e": "E.mp4",
            "eat": "Eat.mp4",
            "engineer": "Engineer.mp4",
            "f": "F.mp4",
            "fight": "Fight.mp4",
            "finish": "Finish.mp4",
            "from": "From.mp4",
            "g": "G.mp4",
            "glitter": "Glitter.mp4",
            "go": "Go.mp4",
            "god": "God.mp4",
            "gold": "Gold.mp4",
            "good": "Good.mp4",
            "great": "Great.mp4",
            "h": "H.mp4",
            "hand": "Hand.mp4",
            "hands": "Hands.mp4",
            "happy": "Happy.mp4",
            "hello": "Hello.mp4",
            "help": "Help.mp4",
            "her": "Her.mp4",
            "here": "Here.mp4",
            "his": "His.mp4",
            "home": "Home.mp4",
            "homepage": "Homepage.mp4",
            "how": "How.mp4",
            "i": "I.mp4",
            "invent": "Invent.mp4",
            "it": "It.mp4",
            "j": "J.mp4",
            "k": "K.mp4",
            "keep": "Keep.mp4",
            "l": "L.mp4",
            "language": "Language.mp4",
            "laugh": "Laugh.mp4",
            "learn": "Learn.mp4",
            "m": "M.mp4",
            "me": "ME.mp4",
            "more": "More.mp4",
            "my": "My.mp4",
            "n": "N.mp4",
            "name": "Name.mp4",
            "next": "Next.mp4",
            "not": "Not.mp4",
            "now": "Now.mp4",
            "o": "O.mp4",
            "of": "Of.mp4",
            "on": "On.mp4",
            "our": "Our.mp4",
            "out": "Out.mp4",
            "p": "P.mp4",
            "pretty": "Pretty.mp4",
            "q": "Q.mp4",
            "r": "R.mp4",
            "right": "Right.mp4",
            "s": "S.mp4",
            "sad": "Sad.mp4",
            "safe": "Safe.mp4",
            "see": "See.mp4",
            "self": "Self.mp4",
            "sign": "Sign.mp4",
            "sing": "Sing.mp4",
            "so": "So.mp4",
            "sound": "Sound.mp4",
            "stay": "Stay.mp4",
            "study": "Study.mp4",
            "t": "T.mp4",
            "talk": "Talk.mp4",
            "television": "Television.mp4",
            "thank you": "Thank You.mp4",
            "thank": "Thank.mp4",
            "that": "That.mp4",
            "they": "They.mp4",
            "this": "This.mp4",
            "those": "Those.mp4",
            "time": "Time.mp4",
            "to": "To.mp4",
            "type": "Type.mp4",
            "u": "U.mp4",
            "us": "Us.mp4",
            "v": "V.mp4",
            "w": "W.mp4",
            "walk": "Walk.mp4",
            "wash": "Wash.mp4",
            "way": "Way.mp4",
            "we": "We.mp4",
            "welcome": "Welcome.mp4",
            "what": "What.mp4",
            "when": "When.mp4",
            "where": "Where.mp4",
            "which": "Which.mp4",
            "who": "Who.mp4",
            "whole": "Whole.mp4",
            "whose": "Whose.mp4",
            "why": "Why.mp4",
            "will": "Will.mp4",
            "with": "With.mp4",
            "without": "Without.mp4",
            "words": "Words.mp4",
            "work": "Work.mp4",
            "world": "World.mp4",
            "wrong": "Wrong.mp4",
            "x": "X.mp4",
            "y": "Y.mp4",
            "you": "You.mp4",
            "your": "Your.mp4",
            "yourself": "Yourself.mp4",
            "z": "Z.mp4"
        }
        
        # Setup directories
        self.video_directory = Path(video_directory)
        self.output_directory = Path(output_directory)
        
        # Create output directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        if not self.video_directory.exists():
            raise FileNotFoundError(f"Video directory '{video_directory}' not found")
            
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError("Spacy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")

    def extract_sov(self, sentence):
        """Extract Subject-Object-Verb components from the sentence"""
        doc = self.nlp(sentence)
        svo_triples = list(textacy.extract.subject_verb_object_triples(doc))
        
        subjects = []
        objects = []
        verbs = []
        other_words = []
        
        if svo_triples:
            for triple in svo_triples:
                if triple[0]:
                    subjects.extend(str(triple[0]).lower().split())
                if triple[2]:
                    objects.extend(str(triple[2]).lower().split())
                if triple[1]:
                    verbs.extend(str(triple[1]).lower().split())
        
        # Handle words not in SVO triples
        for token in doc:
            word = token.text.lower()
            if (word not in subjects and 
                word not in objects and 
                word not in verbs and 
                not token.is_punct and 
                not token.is_space):
                other_words.append(word)
        
        return subjects + objects + verbs + other_words

    def process_sentence(self, sentence):
        """Process a sentence and return a list of video files to process"""
        try:
            words = self.extract_sov(sentence)
            print(f"Words in SOV order: {words}")
            
            video_sequence = []
            for word in words:
                if word in self.sign_word_videos:
                    video_sequence.append(self.sign_word_videos[word])
                else:
                    video_sequence.extend(self.fingerspell(word))
                    
            return video_sequence
            
        except Exception as e:
            print(f"Error processing sentence: {str(e)}")
            return []

    def fingerspell(self, word):
        """Convert a word to a sequence of letter videos for fingerspelling"""
        return [self.sign_word_videos.get(letter.lower(), f"{letter.upper()}.mp4") 
                for letter in word if letter.isalnum()]

    def concatenate_videos(self, video_files, output_filename):
        """Concatenate multiple videos into a single output video"""
        if not video_files:
            return False
        
        try:
            # Get properties of first video to initialize output video
            first_video = str(self.video_directory / video_files[0])
            cap = cv2.VideoCapture(first_video)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            # Create VideoWriter object
            output_path = str(self.output_directory / output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # Process each video
            for video_file in video_files:
                video_path = str(self.video_directory / video_file)
                if not os.path.exists(video_path):
                    print(f"Warning: Video file not found: {video_path}")
                    continue

                cap = cv2.VideoCapture(video_path)
                
                # Read and write frames
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                
                # Add small pause between words (black frames)
                for _ in range(int(fps * 0.2)):  # 0.2 seconds pause
                    blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    out.write(blank_frame)
                
                cap.release()

            out.release()
            print(f"Video saved successfully: {output_filename}")
            return True

        except Exception as e:
            print(f"Error concatenating videos: {str(e)}")
            return False

    def translate_to_video(self, sentence):
        """Translate a sentence to sign language video"""
        try:
            # Process the sentence
            video_files = self.process_sentence(sentence)
            
            if not video_files:
                print("No translation available for the input sentence.")
                return False
            
            # Generate output filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"isl_translation_{timestamp}.mp4"
            
            # Concatenate videos and save
            success = self.concatenate_videos(video_files, output_filename)
            
            if success:
                print(f"Translation saved to: {self.output_directory / output_filename}")
            return success
            
        except Exception as e:
            print(f"Error in translation: {str(e)}")
            return False

def main():
    try:
        translator = ISLTranslator()
        
        print("Indian Sign Language Translator")
        print("Sentences will be converted to Subject-Object-Verb order")
        print(f"Output videos will be saved in: {translator.output_directory}")
        print("Enter 'quit' to exit")
        
        while True:
            sentence = input("\nEnter a sentence to translate: ").strip()
            
            if sentence.lower() == 'quit':
                break
                
            if not sentence:
                print("Please enter a valid sentence.")
                continue
                
            print("Processing:", sentence)
            translator.translate_to_video(sentence)
                
    except KeyboardInterrupt:
        print("\nTranslator terminated by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()