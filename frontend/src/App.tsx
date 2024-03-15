import { useState } from 'react'
import './App.css'
import InputBox from './InputBox'
import OutputBox from './OutputBox'
import axios from 'axios'

interface ApiBody {
  image_bytes: string
}

interface ApiResponse {
  image_bytes: string;
}

function App() {
  const [count, setCount] = useState(0)

  const [inputImage, setInputImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);

  const API_PATH = "http://localhost:8000/api/v1/unet";

  async function handleConvert() {
    if (inputImage) {

      // Extract base64 body from image
      const imgBytes = inputImage.split(',')[1]

      const body: ApiBody = {
        image_bytes: imgBytes
      }

      const response = await axios.post(API_PATH, body, {
        headers: { 'Content-Type': 'application/json' },
      });

      const data: ApiResponse = response.data;
      const newImage = "data:image/jpeg;base64," + data.image_bytes;
      setResultImage(newImage);
    }
  }

  function handleUploadedImage(image: string) {
    setInputImage(image);
  }

  return (
    <main>
      <InputBox onSendData={handleUploadedImage} />
      <div>
        <button
          id="btnConvert"
          className="btn-convert"
          onClick={handleConvert}>
          Convert
        </button>
      </div>
      <OutputBox image={resultImage} />
    </main>
  )
}

export default App
