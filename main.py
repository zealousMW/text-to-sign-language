import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
import json
import cv2
import time

api_key = "AIzaSyDsTNYvMqYM-LAGUd8fB12rWzVixDsU914"
os.environ["API_KEY"] = api_key
genai.configure(api_key=os.environ["API_KEY"])

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_schema": content.Schema(
    type = content.Type.OBJECT,
    properties = {
      "output": content.Schema(
        type = content.Type.ARRAY,
        items = content.Schema(
          type = content.Type.STRING,
        ),
      ),
    },
  ),
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "Indian Sign Language (ISL) Conversion Task\nYou are an expert in Indian Sign Language (ISL) and natural language processing. Your task is to convert English sentences into a sequence of ISL sign words.\nInput\nYou will receive an English sentence.\nOutput\nProvide a list of ISL sign words that best represent the meaning of the input sentence. Each sign word should be separated by a comma.\nGuidelines:\n\nUse standard ISL sign words where possible.\nConsider ISL grammar and syntax, which may differ from English:\n\nISL often uses a Subject-Object-Verb (SOV) word order.\nTime indicators are usually placed at the beginning of sentences.\nAdjectives typically come after the noun they modify.\n\n\nSome English words may not have direct ISL equivalents. In such cases:\n\nUse conceptually accurate signs that convey the meaning.\nIf necessary, break down complex concepts into simpler sign combinations.\n\n\nOmit articles (a, an, the) unless they are crucial for meaning.\nFor proper nouns or specialized terms without standard signs\n\nExamples:\nInput: \"The cat is sleeping on the mat.\"\nOutput: MAT, CAT, SLEEP\nInput: \"What is your name?\"\nOutput: YOUR, NAME, WHAT\nInput: \"I will go to the market tomorrow.\"\nOutput: TOMORROW, MARKET, I, GO\nInput: \"Albert Einstein was a famous physicist.\"\nOutput: ALBERT, EINSTEIN, FAMOUS, PHYSICIST\nRemember, the goal is to provide a sequence of ISL sign words that accurately conveys the meaning of the English sentence in a way that is natural and fluent for ISL users.",
      ],
    },
  ]
)




sign_word_videos = {
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
video_directory = "video"
def simulate_llm_output( sentence):
        words = sentence.lower().split()
        response = chat_session.send_message(words)
        
        return json.dumps(response.text)
def process_llm_output( llm_output):
    try:
        print("Raw LLM Output:", llm_output)
        print("Type of LLM Output:", type(llm_output))

        llm_output = llm_output.strip()
        data = json.loads(llm_output)
        result = []
        # Iterate through the list of words in the "output" array
        print(data[0]["output"])
        for word in data["output"]:
            video = sign_word_videos.get(word.lower())
            if video:
                result.append(video)
            else:
                # If the word is not found, fingerspell it
                result.extend(fingerspell(word))
        return result
    except json.JSONDecodeError:
        print("Error: Invalid JSON from LLM")
        return []
    except KeyError:
        print("Error: Unexpected JSON structure from LLM")
        return []

def fingerspell(word):
        return [sign_word_videos.get(letter.lower(), f"Unknown_{letter}.mp4") for letter in word]

def play_videos( video_files):
        for video_file in video_files:
            video_path = f"{video_directory}/{video_file}"
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Error opening video file {video_file}")
                continue

            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cv2.imshow('ISL Video', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break

            cap.release()
            time.sleep(0.5)  # Pause between videos

        cv2.destroyAllWindows()


test_sentences = [
    "Hello, how are you?",
    "I am learning Indian Sign Language",
    "The quick brown fox jumps over the lazy dog",
    "Python is fun"
]

for sentence in test_sentences:
    print(f"Converting: {sentence}")
    llm_output = simulate_llm_output(sentence)
    video_files = process_llm_output(llm_output)
    print(f"Video sequence: {video_files}")
    play_videos(video_files)
    print("Finished playing. Press any key to continue...")
    input()