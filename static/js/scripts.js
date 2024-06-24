document.getElementById('transcribeForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    const formData = new FormData(this);
    document.getElementById('loading').style.display = 'block';
    document.getElementById('response').style.display = 'none';

    try {
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        document.getElementById('loading').style.display = 'none';

        if (response.ok) {
            document.getElementById('transcription').textContent = result.transcription;
            document.getElementById('summary').textContent = result.summary;
            document.getElementById('result').style.display = 'block';
        } else {
            document.getElementById('response').textContent = result.error;
            document.getElementById('response').classList.add('error');
            document.getElementById('response').style.display = 'block';
        }
    } catch (error) {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('response').textContent = 'An error occurred while processing your request.';
        document.getElementById('response').classList.add('error');
        document.getElementById('response').style.display = 'block';
    }
});
