import { useState } from 'react';

function VideoPlayer() {
  const [selectedContainers, setSelectedContainers] = useState(Array(data.length).fill(false));

  // Existing code here

  fetch('http://localhost:8000/video-data/8')
    .then(response => response.json())
    .then(data => {
      const faceImageContainers = document.querySelector('.face-image-containers');
      for (let i = 0; i < data.length; i++) {
        const faceImageURL = data[i].face;
        const faceImageContainer = document.createElement('div');
        faceImageContainer.classList.add('face-image-container');
        if (selectedContainers[i]) {
          faceImageContainer.classList.add('selected');
        }
        faceImageContainer.setAttribute('data-index', i);

        const faceImage = document.createElement('img');
        faceImage.classList.add('face-image');
        faceImage.setAttribute('src', 'data:image/png;base64,' + faceImageURL);
        faceImageContainer.appendChild(faceImage);

        faceImageContainer.addEventListener('click', function() {
          const index = this.getAttribute('data-index');
          const newSelectedContainers = [...selectedContainers];
          newSelectedContainers[index] = !newSelectedContainers[index];
          setSelectedContainers(newSelectedContainers);
        });

        faceImageContainers.appendChild(faceImageContainer);
      }
    })
    .catch(error => {
      console.error('Error fetching data:', error);
    });

  // Existing code here
}

export default VideoPlayer;
