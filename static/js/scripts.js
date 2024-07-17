document.getElementById("transcribeForm").addEventListener("submit", function(event) {
    event.preventDefault();

    var form = event.target;
    var formData = new FormData(form);
    var xhr = new XMLHttpRequest();

    // Display the processing message before sending the request
    document.getElementById("processing").style.display = "block";
    document.getElementById("processing").textContent = "Processing...";

    xhr.open("POST", form.action, true);

    xhr.onreadystatechange = function() {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            document.getElementById("processing").style.display = "none";

            var response = JSON.parse(xhr.responseText);

            if (xhr.status === 200) {
                document.getElementById("summary-content").textContent = response.summary;
                document.getElementById("transcription-content").textContent = response.transcription;
            } else {
                document.getElementById("processing").style.display = "block";
                document.getElementById("processing").textContent = response.error;
            }

            document.getElementById("youtube-url").disabled = false;
            document.getElementById("file-input").disabled = false;
        }
    };

    xhr.send(formData);

    document.getElementById("youtube-url").disabled = true;
    document.getElementById("file-input").disabled = true;
    document.getElementById("summary-content").textContent = "Summary content will appear here...";
    document.getElementById("transcription-content").textContent = "Transcription content will appear here...";
});

document.getElementById('clear-btn').addEventListener('click', function() {
    document.getElementById('youtube-url').value = '';
    document.getElementById('file-input').value = '';
    document.getElementById("summary-content").textContent = "Summary content will appear here...";
    document.getElementById("transcription-content").textContent = "Transcription content will appear here...";
    document.getElementById("processing").style.display = "none";
    document.getElementById("processing").textContent = "Processing...";
});

document.getElementById('copy-summary-btn').addEventListener('click', function() {
    copyToClipboard('summary-content');
});

document.getElementById('copy-transcription-btn').addEventListener('click', function() {
    copyToClipboard('transcription-content');
});

function copyToClipboard(elementId) {
    var text = document.getElementById(elementId).textContent;
    var textarea = document.createElement("textarea");
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
    alert("Copied to clipboard");
}
