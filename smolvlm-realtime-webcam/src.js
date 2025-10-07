

const video = document.getElementById('videoFeed');
        const canvas = document.getElementById('canvas');
        const baseURL = document.getElementById('baseURL');
        const llmURL = document.getElementById('llmURL');
        const instructionText = document.getElementById('instructionText');
        const smolvlmResponse = document.getElementById('smolvlmResponse');
        const tinyLlamaResponse = document.getElementById('tinyLlamaResponse');
        const phi3Response = document.getElementById('phi3Response');
        const intervalSelect = document.getElementById('intervalSelect');

        const startButton = document.getElementById('startButton');

        instructionText.value = "Where's my water bottle?";

        let stream, intervalId, isProcessing = false;

        async function button(){
            if(but.value === "start"){
                but.value = "stop"
                but.style="background-color: red"
                createStream();
            }
            else if(but.value === "stop"){
                but.value = "start"
                but.style="background-color: rgb(115, 255, 0)"
                stopRecording();
            }
        
        }

        async function input(){
            if(document.getElementById("instructionType").value === "audio"){
                document.getElementById("recordButton").style.display = 'inline';
                document.getElementById("instructionForText").style.display = 'none';                
                console.log("Audio Input")
            }else if(document.getElementById("instructionType").value === "text"){
                document.getElementById("recordButton").style.display = 'none';                
                document.getElementById("instructionForText").style.display = 'inline';                
                console.log("Text input")
            }
}

        async function sendSmolVLMRequest(instruction, imageBase64URL) {
            const response = await fetch(`${baseURL.value}/v1/chat/completions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    max_tokens: 150,
                    messages: [
                        { role: 'user', content: [
                            { type: 'text', text: instruction },
                            { type: 'image_url', image_url: { url: imageBase64URL } }
                        ]}
                    ]
                })
            });
            const data = await response.json();
            return data.choices[0].message.content;
        }

        async function sendLlmRequest(instruction, smolResponse) {
            const finalPrompt = `The user asked: "${instruction}". 
        The vision model responded: "${smolResponse}". 
        Reformulate this answer to better address the user's question. If the user asks about a 
        specific object and the vision model mentions it at all, tell the user it's in view.`;

            const response = await fetch(`${llmURL.value}/v1/completions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: finalPrompt,
                    max_tokens: 50,
                    temperature: 0.7
                })
            });
            const data = await response.json();
            return data.choices[0].text;
        }

        function captureImage() {
            if (!stream || !video.videoWidth) return null;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/jpeg', 0.8);
        }

        async function sendData() {
            if (!isProcessing) return;

            const instruction = instructionText.value;
            const imageBase64URL = captureImage();
            if (!imageBase64URL) return;

            try {
                // 1. Get SmolVLM vision response
                const smolResponse = await sendSmolVLMRequest(instruction, imageBase64URL);
                smolvlmResponse.value = smolResponse;

                // 2. Send result to selected LLM (Phi-3 or TinyLlama)
                if (llmURL.value.includes("8081")) {
                    const tinyResponse = await sendLlmRequest(instruction, smolResponse);
                    tinyLlamaResponse.value = tinyResponse;
                } else if (llmURL.value.includes("8082")) {
                    const phiResponse = await sendLlmRequest(instruction, smolResponse);
                    phi3Response.value = phiResponse;
                }
            } catch (error) {
                console.error("Error in sendData:", error);
                smolvlmResponse.value = "Error: " + error.message;
            }
        }


        async function initCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                smolvlmResponse.value = "Camera ready.";
            } catch (err) {
                alert("Camera access denied: " + err.message);
            }
        }

        function handleStart() {
            if (!stream) return alert("Camera not available.");
            isProcessing = true;
            startButton.textContent = "Stop";
            startButton.classList.replace('start', 'stop');
            instructionText.disabled = true;
            intervalSelect.disabled = true;

            if(intervalSelect.value==="single")
            {
                sendData().then(() => handleStop());
            }
            else
            {
                const intervalMs = parseInt(intervalSelect.value, 10);
                sendData();
                intervalId = setInterval(sendData, intervalMs);
            }
        }

        function handleStop() {
            isProcessing = false;
            clearInterval(intervalId);
            intervalId = null;
            startButton.textContent = "Start";
            startButton.classList.replace('stop', 'start');
            instructionText.disabled = false;
            intervalSelect.disabled = false;
        }




        startButton.addEventListener('click', () => isProcessing ? handleStop() : handleStart());
        window.addEventListener('DOMContentLoaded', initCamera);


        