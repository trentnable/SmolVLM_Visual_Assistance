# text to speech
from io import BytesIO
from gtts import gTTS
import subprocess

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

buf = BytesIO()
gTTS(test, lang="en", tld="co.uk").write_to_fp(buf)
buf.seek(0)

p = subprocess.Popen(
    [r"C:\Users\lmgre\Documents\SIU\Senior Design\pranjal_repo\SmolVLM_Visual_Assistance\FFmpeg\ffplay.exe", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
    stdin=subprocess.PIPE,
)

p.stdin.write(buf.read())
p.stdin.close()
p.wait()