document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            const tabId = button.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Text generation form submission
    const textForm = document.getElementById('text-form');
    if (textForm) {
        textForm.addEventListener('submit', function(e) {
            e.preventDefault();
            generateFromText();
        });
    }

    // Image upload and generation
    const imageDropzone = document.getElementById('image-dropzone');
    const imageInput = document.getElementById('image-input');
    const imagePreview = document.getElementById('image-preview');
    const generateImageButton = document.getElementById('generate-image-button');

    if (imageDropzone && imageInput) {
        // Handle drag and drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            imageDropzone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            imageDropzone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            imageDropzone.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            imageDropzone.classList.add('dropzone-active');
        }

        function unhighlight() {
            imageDropzone.classList.remove('dropzone-active');
        }

        imageDropzone.addEventListener('drop', handleImageDrop, false);

        function handleImageDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;

            if (files.length) {
                imageInput.files = files;
                previewImage(files[0]);
                generateImageButton.disabled = false;
            }
        }

        // Handle file input change
        imageInput.addEventListener('change', function() {
            if (this.files.length) {
                previewImage(this.files[0]);
                generateImageButton.disabled = false;
            }
        });

        // Click on dropzone to trigger file input
        imageDropzone.addEventListener('click', function() {
            imageInput.click();
        });

        // Preview the selected image
        function previewImage(file) {
            if (!file.type.match('image.*')) {
                showError('Please select an image file');
                return;
            }

            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        // Image generation form submission
        const imageForm = document.getElementById('image-form');
        if (imageForm) {
            imageForm.addEventListener('submit', function(e) {
                e.preventDefault();
                generateFromImage();
            });
        }
    }

    // Audio upload and generation
    const audioDropzone = document.getElementById('audio-dropzone');
    const audioInput = document.getElementById('audio-input');
    const audioPlayer = document.getElementById('audio-preview-player');
    const generateAudioButton = document.getElementById('generate-audio-button');

    if (audioDropzone && audioInput) {
        // Handle drag and drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            audioDropzone.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            audioDropzone.addEventListener(eventName, function() {
                audioDropzone.classList.add('dropzone-active');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            audioDropzone.addEventListener(eventName, function() {
                audioDropzone.classList.remove('dropzone-active');
            }, false);
        });

        audioDropzone.addEventListener('drop', function(e) {
            const dt = e.dataTransfer;
            const files = dt.files;

            if (files.length) {
                audioInput.files = files;
                previewAudio(files[0]);
                generateAudioButton.disabled = false;
            }
        }, false);

        // Handle file input change
        audioInput.addEventListener('change', function() {
            if (this.files.length) {
                previewAudio(this.files[0]);
                generateAudioButton.disabled = false;
            }
        });

        // Click on dropzone to trigger file input
        audioDropzone.addEventListener('click', function() {
            audioInput.click();
        });

        // Preview the selected audio
        function previewAudio(file) {
            if (!file.type.match('audio.*')) {
                showError('Please select an audio file');
                return;
            }

            const reader = new FileReader();
            reader.onload = function(e) {
                audioPlayer.src = e.target.result;
                audioPlayer.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        // Audio generation form submission
        const audioForm = document.getElementById('audio-form');
        if (audioForm) {
            audioForm.addEventListener('submit', function(e) {
                e.preventDefault();
                generateFromAudio();
            });
        }
    }

    // Generation functions
    function generateFromText() {
        const textPrompt = document.getElementById('text-prompt').value.trim();
        if (!textPrompt) {
            showError('Please enter a text prompt', 'text-error');
            return;
        }

        // Show loading indicator
        document.getElementById('text-loading').style.display = 'block';
        document.getElementById('text-error').style.display = 'none';
        document.getElementById('text-result').style.display = 'none';

        console.log('Generating music from text: ' + textPrompt);

        // Create form data
        const formData = new FormData();
        formData.append('text_prompt', textPrompt);

        // Add music parameters if they exist
        const tempo = document.getElementById('text-tempo');
        const mood = document.getElementById('text-mood');
        const genre = document.getElementById('text-genre');

        if (tempo) formData.append('tempo', tempo.value);
        if (mood) formData.append('mood', mood.value);
        if (genre) formData.append('genre', genre.value);

        // Send request to server
        fetch('/generate-from-text', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Server response:', data);
            document.getElementById('text-loading').style.display = 'none';

            if (data.error) {
                showError(data.error, 'text-error');
                return;
            }

            // Set audio source first
            const audioPlayer = document.getElementById('text-audio-player');
            audioPlayer.src = data.audio_path;

            // Set download link
            document.getElementById('text-download-link').href = '/download/' + data.audio_path.split('/').pop();

            // Force display of result section
            const resultSection = document.getElementById('text-result');
            resultSection.style.display = 'block';
            resultSection.classList.add('active');

            console.log('Audio path:', data.audio_path);
            console.log('Result section display:', resultSection.style.display);

            // Initialize waveform if wavesurfer.js is available
            if (typeof WaveSurfer !== 'undefined') {
                setTimeout(() => {
                    console.log('Initializing waveform...');
                    initWaveform('text-waveform', data.audio_path);
                }, 100); // Small delay to ensure DOM is updated
            } else {
                console.warn('WaveSurfer is not available');
            }
        })
        .catch(error => {
            console.error('Error generating music:', error);
            document.getElementById('text-loading').style.display = 'none';
            showError('An error occurred: ' + error, 'text-error');
        });
    }

    function generateFromImage() {
        const imageInput = document.getElementById('image-input');
        if (!imageInput.files.length) {
            showError('Please select an image', 'image-error');
            return;
        }

        // Show loading indicator
        document.getElementById('image-loading').style.display = 'block';
        document.getElementById('image-error').style.display = 'none';
        document.getElementById('image-result').style.display = 'none';

        // Create form data
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);

        // Add music parameters if they exist
        const tempo = document.getElementById('image-tempo');
        const mood = document.getElementById('image-mood');
        const genre = document.getElementById('image-genre');

        if (tempo) formData.append('tempo', tempo.value);
        if (mood) formData.append('mood', mood.value);
        if (genre) formData.append('genre', genre.value);

        // Send request to server
        fetch('/generate-from-image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('image-loading').style.display = 'none';

            if (data.error) {
                showError(data.error, 'image-error');
                return;
            }

            // Display the result
            document.getElementById('image-result').style.display = 'block';
            document.getElementById('image-audio-player').src = data.audio_path;
            document.getElementById('image-download-link').href = '/download/' + data.audio_path.split('/').pop();

            // Initialize waveform if wavesurfer.js is available
            if (typeof WaveSurfer !== 'undefined') {
                initWaveform('image-waveform', data.audio_path);
            }
        })
        .catch(error => {
            document.getElementById('image-loading').style.display = 'none';
            showError('An error occurred: ' + error, 'image-error');
        });
    }

    function generateFromAudio() {
        const audioInput = document.getElementById('audio-input');
        if (!audioInput.files.length) {
            showError('Please select an audio file', 'audio-error');
            return;
        }

        // Show loading indicator
        document.getElementById('audio-loading').style.display = 'block';
        document.getElementById('audio-error').style.display = 'none';
        document.getElementById('audio-result').style.display = 'none';

        // Create form data
        const formData = new FormData();
        formData.append('audio', audioInput.files[0]);

        // Add music parameters if they exist
        const tempo = document.getElementById('audio-tempo');
        const mood = document.getElementById('audio-mood');
        const genre = document.getElementById('audio-genre');

        if (tempo) formData.append('tempo', tempo.value);
        if (mood) formData.append('mood', mood.value);
        if (genre) formData.append('genre', genre.value);

        // Send request to server
        fetch('/generate-from-audio', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('audio-loading').style.display = 'none';

            if (data.error) {
                showError(data.error, 'audio-error');
                return;
            }

            // Display the result
            document.getElementById('audio-result').style.display = 'block';
            document.getElementById('audio-generated-player').src = data.generated_audio_path;
            document.getElementById('audio-download-link').href = '/download/' + data.generated_audio_path.split('/').pop();

            // Initialize waveform if wavesurfer.js is available
            if (typeof WaveSurfer !== 'undefined') {
                initWaveform('audio-waveform', data.generated_audio_path);
            }
        })
        .catch(error => {
            document.getElementById('audio-loading').style.display = 'none';
            showError('An error occurred: ' + error, 'audio-error');
        });
    }

    // Helper functions
    function showError(message, elementId = null) {
        if (elementId) {
            const errorElement = document.getElementById(elementId);
            if (errorElement) {
                errorElement.textContent = message;
                errorElement.style.display = 'block';
            }
        } else {
            alert(message);
        }
    }

    function initWaveform(containerId, audioSrc) {
        console.log('Initializing waveform for container:', containerId, 'with audio source:', audioSrc);

        const container = document.getElementById(containerId);
        if (!container) {
            console.error('Container not found:', containerId);
            return;
        }

        // Clear previous waveform
        container.innerHTML = '';

        try {
            const wavesurfer = WaveSurfer.create({
                container: '#' + containerId,
                waveColor: '#4895ef',
                progressColor: '#f72585',
                cursorColor: '#4cc9f0',
                barWidth: 2,
                barRadius: 3,
                cursorWidth: 1,
                height: 80,
                barGap: 2
            });

            console.log('WaveSurfer instance created');

            // Handle errors during loading
            wavesurfer.on('error', function(err) {
                console.error('WaveSurfer error:', err);
            });

            // Log when ready
            wavesurfer.on('ready', function() {
                console.log('WaveSurfer is ready');
            });

            // Load the audio file
            wavesurfer.load(audioSrc);
            console.log('Audio loading started');

            // Add play/pause functionality
            const playButton = container.parentElement.querySelector('.play-button');
            if (playButton) {
                playButton.addEventListener('click', function() {
                    console.log('Play button clicked');
                    wavesurfer.playPause();

                    const icon = this.querySelector('i');
                    if (wavesurfer.isPlaying()) {
                        icon.className = 'fas fa-pause';
                    } else {
                        icon.className = 'fas fa-play';
                    }
                });
            } else {
                console.warn('Play button not found');
            }

            // Update time display
            const currentTime = container.parentElement.querySelector('.current-time');
            const totalTime = container.parentElement.querySelector('.total-time');

            if (currentTime && totalTime) {
                wavesurfer.on('ready', function() {
                    const duration = wavesurfer.getDuration();
                    totalTime.textContent = formatTime(duration);
                    console.log('Audio duration:', duration);
                });

                wavesurfer.on('audioprocess', function() {
                    const time = wavesurfer.getCurrentTime();
                    currentTime.textContent = formatTime(time);
                });
            } else {
                console.warn('Time display elements not found');
            }

            // Reset play button when playback ends
            wavesurfer.on('finish', function() {
                console.log('Playback finished');
                if (playButton) {
                    const icon = playButton.querySelector('i');
                    if (icon) {
                        icon.className = 'fas fa-play';
                    }
                }
            });

            return wavesurfer;
        } catch (error) {
            console.error('Error initializing waveform:', error);
        }
    }

    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return minutes + ':' + (remainingSeconds < 10 ? '0' : '') + remainingSeconds;
    }

    // Initialize the first tab
    if (tabButtons.length > 0) {
        tabButtons[0].click();
    }
});
