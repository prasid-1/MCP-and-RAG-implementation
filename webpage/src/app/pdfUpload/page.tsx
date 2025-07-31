"use client";

import { useRef } from "react";
export default function UploadPDF() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      alert("Please select a PDF file to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:3001/upload-pdf", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload PDF file");
      }

      const result = await response.json();
      console.log("Upload successful:", result);
    } catch (error) {
      console.error("Error uploading PDF file:", error);
    }
  };

  return (
    <div>
      <h1>Upload PDF Files</h1>
      <input type="file" accept="application/pdf" ref={fileInputRef} />
      <button onClick={handleUpload}>Upload PDF</button>
    </div>
  );
}
