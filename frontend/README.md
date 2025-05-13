# Eigenfaces Frontend

Vue.js frontend for the Eigenfaces facial recognition project.

## Features (Coming Soon)

- User interface for uploading and processing face images
- Face recognition functionality
- Visualization of eigenfaces and mean face
- Interactive demo of face recognition
- Admin panel for training and managing the system

## Getting Started

This frontend component will be developed in Week 9-10 of the project.

## Planned Technologies

- Vue.js 3 with Composition API
- Vite for build tooling
- Axios for API communication
- Tailwind CSS for styling

## Installation (Coming Soon)

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Communication with Backend

The frontend will communicate with the Django backend through RESTful API endpoints:

- `POST /api/faces/process/`: Upload and process face images
- `POST /api/faces/recognize/`: Submit images for recognition
- `GET /api/faces/visualize/`: Retrieve eigenfaces visualization 