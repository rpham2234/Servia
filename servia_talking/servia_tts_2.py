import speech_recognition as sr
from openai import OpenAI
import pyaudio
import pyttsx3

#Initializing stuff

engine = pyttsx3.init()
reader = sr.Recognizer()
client = OpenAI() #api_key="paste api key here"

print("Ask me any question")

while True:
    #Listens to user talk from microphone
    try:
        with sr.Microphone(device_index=3) as source: #try either 1, 3 or 23. Check output.txt for more details. 3 works best though
            audio_data = reader.listen(source)
            print("Understanding your question...")
            question = reader.recognize_google(audio_data)
            
            print(question)

        #Sends user speech to ChatGPT
        print("Waiting for response...")
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": str(question)}
        ]
        )

        
        response = str(completion.choices[0].message.content)

        #Says response to User
        print(response)
        engine.say(response)
        engine.runAndWait()

    #If TTS engine does not get input, errors would be thrown. This is what to do if there is that error
    except:
        response = "Sorry, I didn't get that, please speak again"
        print(response)
        engine.say(response)
        engine.runAndWait()
        
    print("Ask another question")
