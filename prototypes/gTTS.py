from gtts import gTTS
from playsound import playsound
import tempfile

test = "The quick brown fox jumps over the lazy dog."
select = 0

# Generate speech

if select == 0:
    print("british")
    tts = gTTS(test, lang="en", tld='co.uk')

elif select == 1:
    print("irish")
    tts = gTTS(test, lang="en", tld='ie')

elif select == 2:
    print("australian")
    tts = gTTS(test, lang="en", tld='com.au')

elif select == 3:
    print("USA")
    tts = gTTS(test, lang="en", tld='us')

elif select == 4:
    print("france")
    tts = gTTS(test, lang="en", tld='fr')

elif select == 5:
    print("spain")
    tts = gTTS(test, lang="en", tld='es')

# Save to a temporary mp3 file
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
    temp_path = fp.name
    tts.save(temp_path)

# Play it
playsound(temp_path)
