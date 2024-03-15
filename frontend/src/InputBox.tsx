import React, { FormEventHandler, useState } from "react";

interface ChildProps {
  onSendData: (data: string) => void;
}

function InputBox(props: ChildProps) {
  const [image, setImage] = useState<string | null>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];

    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (reader.result) {
          const res = reader.result as string
          setImage(res);
          props.onSendData(res);
        }
      }
      reader.readAsDataURL(file);
    }
  }

  function handleButtonClick(e: any) {
    e.preventDefault();
    document.getElementById("file-upload")?.click();
  }

  return (
    <section>
      <div className="box-capture">Original image</div>
      <div className="box box-input" style={{
        backgroundImage: image ? `url(${image})` : 'none',
      }}>
        <input id="file-upload"
          accept="image/jpeg"
          type="file"
          onChange={handleFileChange} />
        <button className="btn-upload" onClick={handleButtonClick}>
          Upload file
        </button>
      </div>
    </section>
  )
}

export default InputBox;